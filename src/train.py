import time
from typing import Any, Callable, Tuple
import cnnmodel
import unetmodel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class GANProject(nn.Module):
    def __init__(self, load_unet=None, load_cnn=None, load_branch=None, img_size:int = 128, debug:bool = False, small:bool = False, lambd:float = 0.2, activation=nn.ReLU) -> None:
        super(GANProject, self).__init__()

        self.img_size = img_size
        self.debug = debug
        self.lambd = lambd
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if load_unet is None:
            if small:
                self.generator = unetmodel.UNetAutoSmall(3, activation).to(self.device)
            else:
                self.generator = unetmodel.UNetAuto(3, activation).to(self.device)
        else:
            self.generator = load_unet.to(self.device)
        if load_cnn is None:
            self.discriminator = cnnmodel.CNNModel(3, img_size, activation).to(self.device)
        else:
            self.discriminator = load_cnn.to(self.device)
        if load_branch is None:
            self.branch = unetmodel.BranchEncoder(3, activation).to(self.device)
        else:
            self.branch = load_branch.to(self.device)
        self.opt_Gen = torch.optim.Adam(self.generator.parameters(), lr=0.005)
        self.opt_Dsc = torch.optim.Adam(self.discriminator.parameters(), lr=0.005)
        self.opt_Brc = torch.optim.Adam(self.discriminator.parameters(), lr=0.005)
    
    def train_model(self, trainLoader: DataLoader, validLoader: DataLoader, metric:str='loss', epochs:int = 10):

        metrics = {'train' : {'rec_loss' : [], 'gen_loss' : [], 'd_loss' : [], 'f_loss' : []},
				'valid' : {'rec_loss' : [], 'gen_loss' : [], 'd_loss' : []}}
        
        best_metric = 0.1
        
        for epoch in range(epochs):
            start = time.time()
            t_metrics = self.ganTrain_epoch(trainLoader)
            v_metrics = self.ganEval_epoch(validLoader)
            delta_t = time.time() - start

            metrics['train']['rec_loss'].append(t_metrics['rec_loss'])
            metrics['train']['gen_loss'].append(t_metrics['gen_loss'])
            metrics['train']['d_loss'].append(t_metrics['d_loss'])
            metrics['train']['f_loss'].append(t_metrics['feature_loss'])
            metrics['valid']['rec_loss'].append(v_metrics['rec_loss'])
            metrics['valid']['gen_loss'].append(v_metrics['gen_loss'])
            metrics['valid']['d_loss'].append(v_metrics['d_loss'])

            print(f'Epoch {epoch} Time: {delta_t:.2f}s')
            print('Train: ', t_metrics)
            print('Validation: ', v_metrics)

            if v_metrics['rec_loss'] < best_metric:
                torch.save(self.generator, f"genproject_e{epoch}.pt")
                torch.save(self.discriminator, f"dscproject_e{epoch}.pt")
                torch.save(self.branch, f'brcproject_e{epoch}.pt')
                best_metric = v_metrics['rec_loss']

    
    def ganTrain_epoch(self, trainLoader: DataLoader):
        self.generator.train()
        self.discriminator.train()
        self.branch.train()

        t_metrics = {'rec_loss' : 0, 'gen_loss' : 0, 'd_loss' : 0, 'feature_loss' : 0}

        for img, _ in trainLoader:
            img_real = self.gaussian(img,0.2).to(self.device)
            img_fake, intermediates = self.generator.yield_forward(img_real)
            r_intermediates = self.branch.yield_forward(img_fake)

            self.discriminator.zero_grad()

            d_loss = self.dsc_loss(img_real, img_fake)
            d_loss.backward()
            self.opt_Dsc.step()

            self.generator.zero_grad()

            r_metrics = self.gen_loss(img_real, img_fake, self.lambd)
            r_metrics['L_r'].backward(retain_graph=True)
            #rec_loss = self.rec_loss(img_real, img_fake)
            #rec_loss.backward()

            self.branch.zero_grad()

            feature_loss = self.feature_loss(r_intermediates, intermediates)
            feature_loss.backward()
            self.opt_Gen.step()
            self.opt_Brc.step()

            #t_metrics['rec_loss'] += r_metrics['rec_loss']
            #t_metrics['gen_loss'] += r_metrics['gen_loss']
            t_metrics['rec_loss'] += r_metrics['rec_loss']
            t_metrics['gen_loss'] += r_metrics['gen_loss']
            t_metrics['d_loss'] += d_loss
            t_metrics['feature_loss'] += feature_loss

        t_metrics['rec_loss'] = t_metrics['rec_loss'].item() / ((len(trainLoader.dataset)) / trainLoader.batch_size)
        t_metrics['gen_loss'] = t_metrics['gen_loss'].item() / ((len(trainLoader.dataset)) / trainLoader.batch_size)
        t_metrics['d_loss'] = t_metrics['d_loss'].item() / ((len(trainLoader.dataset)) / trainLoader.batch_size)
        t_metrics['feature_loss'] = t_metrics['feature_loss'].item() / ((len(trainLoader.dataset)) / trainLoader.batch_size)
        return t_metrics
    
    def ganEval_epoch(self, validLoader: DataLoader):
        self.generator.eval()
        self.discriminator.eval()
        self.branch.eval()

        v_metrics = {'rec_loss' : 0, 'gen_loss' : 0, 'd_loss' : 0}

        with torch.no_grad():
            for img, _ in validLoader:
                img_real = img.to(self.device)
                img_fake = self.generator(img_real)

                d_loss = self.dsc_loss(img_real, img_fake)
                r_metrics = self.gen_loss(img_real, img_fake, self.lambd)

                v_metrics['rec_loss'] += r_metrics['rec_loss']
                v_metrics['gen_loss'] += r_metrics['gen_loss']
                v_metrics['d_loss'] += d_loss

        v_metrics['rec_loss'] = v_metrics['rec_loss'].item() / ((len(validLoader.dataset)) / validLoader.batch_size)
        v_metrics['gen_loss'] = v_metrics['gen_loss'].item() / ((len(validLoader.dataset)) / validLoader.batch_size)
        v_metrics['d_loss'] = v_metrics['d_loss'].item() / ((len(validLoader.dataset)) / validLoader.batch_size)
        return v_metrics
    
    def evaluate(self, loader: DataLoader) -> dict:
        self.discriminator.eval()
        predictions = []
        ground_truth = []
        with torch.no_grad():
            total = 0
            correct = 0
            for img, label in loader:
                img = img.to(self.device)
                label = label.to(self.device)
                output = self.discriminator(img)
                total += img.size(0)
                _, preds = torch.max(output, 1)
                predictions.extend(preds.cpu().numpy())
                ground_truth.extend(label.cpu().numpy())
                correct += (preds == label).sum().item()
            accuracy = correct / total
        return {'accuracy':accuracy, 'gt':ground_truth, 'pred':predictions}

    def gen_loss(self, x_real: torch.Tensor, x_fake: torch.Tensor, lambd: float) -> dict:

        pred = self.discriminator(x_fake)
        y = torch.ones_like(pred)

        rec_loss = F.mse_loss(x_fake, x_real)
        gen_loss = F.binary_cross_entropy_with_logits(pred, y)

        L_r = gen_loss + lambd * rec_loss

        return {'rec_loss':rec_loss, 'gen_loss':gen_loss, 'L_r':L_r}
    
    def dsc_loss(self, x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:

        pred_r = self.discriminator(x_real)
        pred_f = self.discriminator(x_fake.detach())

        y_r = torch.ones_like(pred_r)
        y_f = torch.zeros_like(pred_f)

        r_loss = F.binary_cross_entropy_with_logits(pred_r, y_r)
        f_loss = F.binary_cross_entropy_with_logits(pred_f, y_f)

        return r_loss + f_loss
    
    def gen_Wloss(self, x_real: torch.Tensor, x_fake: torch.Tensor, lambd: float) -> dict:

        pred = torch.sigmoid(self.discriminator(x_fake))

        rec_loss = F.mse_loss(x_fake, x_real)
        gen_loss = -torch.mean(pred)

        L_r = gen_loss + lambd * rec_loss

        return {'rec_loss':rec_loss, 'gen_loss':gen_loss, 'L_r':L_r}
    
    def dsc_Wloss(self, x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:

        pred_r = torch.sigmoid(self.discriminator(x_real))
        pred_f = torch.sigmoid(self.discriminator(x_fake.detach()))

        wd_loss = -torch.mean(pred_r) + torch.mean(pred_f)

        return wd_loss
    
    def rec_loss(self, x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:
        out = F.mse_loss(x_fake.detach(), x_real)
        return out
    
    def feature_loss(self, encoder_int, reflection_int, lambds=[0.2,0.4]):
        layer1_loss = F.mse_loss(encoder_int[0], reflection_int[0]) * lambds[0]
        layer2_loss = F.mse_loss(encoder_int[1], reflection_int[1]) * lambds[1]
        layer3_loss = F.mse_loss(encoder_int[2], reflection_int[2])
        ec_loss = F.mse_loss(encoder_int[3], reflection_int[3])
        return layer1_loss + layer2_loss + layer3_loss + ec_loss

    def gaussian(self, img:torch.Tensor, stddev):
        if self.training:
            noise = torch.randn_like(img) * stddev
            return img + noise
        return img

    def stat_report(self, gen_loss, dsc_loss, epoch, batch, deltatime) -> None:
        print(f"Epoch {epoch}, batch {batch}; Losses = G:{gen_loss:<10.4f} vs. D:{dsc_loss:<10.4f}; time: {deltatime:<10.1f} sec")

class ForestFireDataset(ImageFolder):
    def __init__(self, root: str, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, loader: Callable[[str], Any] = ..., is_valid_file: Callable[[str], bool] | None = None):
        super(ForestFireDataset, self).__init__(root=root, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if sample.size[1] > 2160:
            apply = v2.Resize(size=2160, max_size=3840)
            sample = apply(sample)
        elif sample.size[1] < 768:
            apply = v2.Resize(size=1080, interpolation=InterpolationMode.BICUBIC)
            sample = apply(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        return (sample, target)


def save_model(model:GANProject, name):
    model.ready_save()
    torch.save(model, name)

#https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
#https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/
#https://github.com/arseniybelkov/Novelty_Detection/blob/master/model.py