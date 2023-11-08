import time
from typing import Any, Callable, Tuple

from torcheval.metrics import BinaryAccuracy
import cnnmodel
import unetmodel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GANProject(nn.Module):
    def __init__(self, load_unet=None, load_cnn=None, load_branch=None, img_size:int = 128, debug:bool = False, small:bool = False, lambd:float = 0.2, activation=nn.ReLU()) -> None:
        super(GANProject, self).__init__()

        self.img_size = img_size
        self.debug = debug
        self.lambd = lambd
        if load_unet is None:
            if small:
                self.generator = unetmodel.UNetAutoSmall(3, activation).to(device)
            else:
                self.generator = unetmodel.UNetAuto(3, activation).to(device)
        else:
            self.generator = load_unet.to(device)
        if load_cnn is None:
            self.discriminator = cnnmodel.CNNModel(3, img_size, activation).to(device)
        else:
            self.discriminator = load_cnn.to(device)
        if load_branch is None:
            self.encoder = unetmodel.Encoder(3, activation).to(device)
        else:
            self.encoder = load_branch.to(device)
        self.opt_Gen = torch.optim.Adam(self.generator.parameters(), lr=0.002)
        self.opt_Dsc = torch.optim.Adam(self.discriminator.parameters(), lr=0.002)
        self.opt_Enc = torch.optim.Adam(self.discriminator.parameters(), lr=0.002)
    
    def train_model(self, trainLoader: DataLoader, validLoader: DataLoader, metric:str='loss', epochs:int = 10, previous:int = 0, eps:float = 0.01):

        metrics = {'train' : {'rec_loss' : [], 'd_loss' : [], 'f_loss' : [], 'ec_loss' : []},
				'valid' : {'rec_loss' : [], 'd_loss' : [], 'ec_loss' : [], 'accuracy' : []}}
        
        best_metric = 0.2
        
        for epoch in range(epochs):
            start = time.time()
            t_metrics = self.ganTrain_epoch(trainLoader)
            v_metrics = self.evaluate(validLoader)
            delta_t = time.time() - start

            metrics['train']['rec_loss'].append(t_metrics['rec_loss'])
            metrics['train']['d_loss'].append(t_metrics['d_loss'])
            metrics['train']['f_loss'].append(t_metrics['feature_loss'])
            metrics['train']['ec_loss'].append(t_metrics['ec_loss'])
            metrics['valid']['rec_loss'].append(v_metrics['rec_loss'])
            metrics['valid']['d_loss'].append(v_metrics['d_loss'])
            metrics['valid']['ec_loss'].append(v_metrics['ec_loss'])
            metrics['valid']['accuracy'].append(v_metrics['accuracy'])

            print(f'Epoch {epoch+previous} Time: {delta_t:.2f}s')
            print('Train: ', t_metrics)
            print('Validation: ', v_metrics)

            if v_metrics['rec_loss'] < best_metric and t_metrics['rec_loss'] < 0.2:
                torch.save(self.generator, f"genproject_e{epoch+previous}.pt")
                torch.save(self.discriminator, f"dscproject_e{epoch+previous}.pt")
                torch.save(self.encoder, f'brcproject_e{epoch+previous}.pt')
                best_metric = v_metrics['rec_loss'] - eps

        if best_metric < eps:
            torch.save(self.generator, f"genproject_e{epoch+previous}.pt")
            torch.save(self.discriminator, f"dscproject_e{epoch+previous}.pt")
            torch.save(self.encoder, f'brcproject_e{epoch+previous}.pt')
            print("Target metric not reached. Saving intermediate.")
    
    def ganTrain_epoch(self, trainLoader: DataLoader):
        self.generator.train()
        self.discriminator.train()
        self.encoder.train()

        t_metrics = {'rec_loss' : 0, 'd_loss' : 0, 'ec_loss':0, 'feature_loss' : 0}

        for img, _ in trainLoader:
            img_real = torch.randn_like(img, device=device) * 0.2 + img.to(device)
            img_fake, gc1,gc2,gc3,gd1 = self.generator(img_real)
            ec1,ec2,ec3,ed1 = self.encoder(img_fake.detach())

            self.generator.zero_grad()
            self.encoder.zero_grad()

            system_loss = self.feature_loss(ec1=ec1, ec2=ec2, ec3=ec3, ed1=ed1, gc1=gc1, gc2=gc2, gc3=gc3, gd1=gd1)
            system_loss.backward()
            self.opt_Enc.step()

            rec_loss = self.gen_loss(img_real, img_fake)
            rec_loss.backward()
            self.opt_Gen.step()

            self.discriminator.zero_grad()
            
            d_loss = self.dsc_Wloss(img_real, img_fake)
            d_loss.backward()
            self.opt_Dsc.step()

            #t_metrics['rec_loss'] += r_metrics['rec_loss']
            #t_metrics['gen_loss'] += r_metrics['gen_loss']
            t_metrics['rec_loss'] += rec_loss
            t_metrics['d_loss'] += d_loss
            t_metrics['feature_loss'] += system_loss

        t_metrics['rec_loss'] = t_metrics['rec_loss'].item() / ((len(trainLoader.dataset)) / trainLoader.batch_size)
        t_metrics['d_loss'] = t_metrics['d_loss'].item() / ((len(trainLoader.dataset)) / trainLoader.batch_size)
        t_metrics['feature_loss'] = t_metrics['feature_loss'].item() / ((len(trainLoader.dataset)) / trainLoader.batch_size)
        return t_metrics
    
    def evaluate(self, loader: DataLoader, lambds=[0.3,0.3,0.3]):
        self.generator.eval()
        self.discriminator.eval()
        self.encoder.eval()
        metric = BinaryAccuracy(threshold=0.5)
        v_metrics = {'rec_loss' : 0, 'd_loss' : 0, 'ec_loss' : 0, 'accuracy' : 0}
        with torch.no_grad():
            total = 0
            for img, label in loader:
                real_img = img.to(device)
                fake_img, _,_,_,gd1 = self.generator(real_img)
                _,_,_,ed1 = self.encoder(fake_img.detach())
                label = label.to(device)

                term1 = F.mse_loss(fake_img, real_img)
                term2 = (1 - self.discriminator(fake_img)) * lambds[1]
                term3 = F.mse_loss(ed1, gd1) * lambds[2]
                d_loss = self.dsc_Wloss(real_img, fake_img)

                total += img.size(0)
                axj = F.normalize(term1*lambds[0] + term2 + term3, dim=0)
                print(axj)
                metric.update(axj.flatten(), label)
                v_metrics['rec_loss'] += term1
                v_metrics['ec_loss'] += term3
                v_metrics['d_loss'] += d_loss

        accuracy = metric.compute().item() / total
        v_metrics['accuracy'] = accuracy
        v_metrics['rec_loss'] = v_metrics['rec_loss'].item() / ((len(loader.dataset)) / loader.batch_size)
        v_metrics['ec_loss'] = v_metrics['ec_loss'].item() / ((len(loader.dataset)) / loader.batch_size)
        v_metrics['d_loss'] = v_metrics['d_loss'].item() / ((len(loader.dataset)) / loader.batch_size)
        return v_metrics
    
    def singleton(self, img):
        img_real = torch.randn_like(img, device=device) * 0.2 + img.to(device)
        img_fake, gc1,gc2,gc3,gd1 = self.generator(img_real)
        ec1,ec2,ec3,ed1 = self.encoder(img_fake.detach())

        self.generator.zero_grad()
        self.encoder.zero_grad()

        system_loss = self.feature_loss(ec1=ec1, ec2=ec2, ec3=ec3, ed1=ed1, gc1=gc1, gc2=gc2, gc3=gc3, gd1=gd1)
        system_loss.backward(retain_graph=True)
        self.opt_Enc.step()

        rec_loss = self.gen_loss(img_real, img_fake)
        rec_loss.backward()
        self.opt_Gen.step()

        self.discriminator.zero_grad()
            
        d_loss = self.dsc_Wloss(img_real, img_fake)
        d_loss.backward()
        self.opt_Dsc.step()

        return system_loss, rec_loss, d_loss

    def gen_loss(self, x_real: torch.Tensor, x_fake: torch.Tensor, lambd: float=0.2):

        #pred = self.discriminator(x_fake)
        #y = torch.ones_like(pred)

        rec_loss = F.mse_loss(x_fake, x_real)
        #rec_loss = F.l1_loss(x_fake, x_real)
        #gen_loss = F.binary_cross_entropy_with_logits(pred, y)

        #L_r = gen_loss + lambd * rec_loss

        return rec_loss #{'rec_loss':rec_loss, 'gen_loss':gen_loss, 'L_r':L_r}
    
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
    
    def feature_loss(self, ec1, ec2, ec3, ed1, gc1, gc2, gc3, gd1, lambds=[0.3,0.4,0.5]):
        layer1_loss = F.mse_loss(ec1, gc1.detach()) * lambds[0]
        layer2_loss = F.mse_loss(ec2, gc2.detach()) * lambds[1]
        layer3_loss = F.mse_loss(ec3, gc3.detach()) * lambds[2]
        ec_loss = F.mse_loss(ed1, gd1.detach())
        return layer1_loss + layer2_loss + layer3_loss + ec_loss

    def gaussian(self, img:torch.Tensor, stddev=0.2):
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
        if self.transform is not None:
            sample = self.transform(sample)

        return (sample, target)


def save_model(model:GANProject, name):
    model.ready_save()
    torch.save(model, name)

#https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
#https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/
#https://github.com/arseniybelkov/Novelty_Detection/blob/master/model.py