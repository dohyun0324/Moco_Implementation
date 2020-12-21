import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.datasets as datasets

import numpy as np

import os

import time
import math
from torchvision.models.resnet import conv3x3

B = 256

from torch.optim.optimizer import Optimizer, required

dataset_path = '../../../home_klimt/dohyun.kim/'
pretrained_path = '../../../home_klimt/dohyun.kim/pretrained_byol_depth_98.pt'

class SGD_with_lars(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    """

    def __init__(self, params, lr=required, momentum=0, weight_decay=0, trust_coef=1.): # need to add trust coef
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if trust_coef < 0.0:
            raise ValueError("Invalid trust_coef value: {}".format(trust_coef))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, trust_coef=trust_coef)

        super(SGD_with_lars, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_with_lars, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            trust_coef = group['trust_coef']
            global_lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                p_norm = torch.norm(p.data, p=2)
                d_p_norm = torch.norm(d_p, p=2).add_(momentum, p_norm)
                lr = torch.div(p_norm, d_p_norm).mul_(trust_coef)

                lr.mul_(global_lr)

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                d_p.mul_(lr)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.data.add_(-1, d_p)

        return loss

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, norm_layer, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        
        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        residual = x 
        residual = self.bn1(residual)
        residual = self.relu1(residual)
        residual = self.conv1(residual)

        residual = self.bn2(residual)
        residual = self.relu2(residual)
        residual = self.conv2(residual)

        if self.downsample is not None:
            x = self.downsample(x)
        return x + residual

class Downsample(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(Downsample, self).__init__()
        self.avg = nn.AvgPool2d(stride)
        assert nOut % nIn == 0
        self.expand_ratio = nOut // nIn

    def forward(self, x):
        x = self.avg(x)
        return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)

class ResNetCifar(nn.Module):
    def __init__(self, depth, width=1, classes=10, channels=3, norm_layer=nn.BatchNorm2d):
        assert (depth - 2) % 6 == 0         # depth is 6N+2
        self.N = (depth - 2) // 6
        super(ResNetCifar, self).__init__()

        # Following the Wide ResNet convention, we fix the very first convolution
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.inplanes = 16
        self.layer1 = self._make_layer(norm_layer, 16 * width)
        self.layer2 = self._make_layer(norm_layer, 32 * width, stride=2)
        self.layer3 = self._make_layer(norm_layer, 64 * width, stride=2)
        self.bn = norm_layer(64 * width)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
    def _make_layer(self, norm_layer, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = Downsample(self.inplanes, planes, stride)
        layers = [BasicBlock(self.inplanes, planes, norm_layer, stride, downsample)]
        self.inplanes = planes
        for i in range(self.N - 1):
            layers.append(BasicBlock(self.inplanes, planes, norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
class DuplicatedCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        img1 = img.copy()
        img2 = img.copy()
        for t in self.transforms:
            img1 = t(img1)
            img2 = t(img2)
        return img1, img2

import cv2
cv2.setNumThreads(0)

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

import torchvision.transforms as transforms

img_size = (96, 96)
'''
#Augmentation 3
color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
train_transform = DuplicatedCompose([
    transforms.RandomResizedCrop(size=(32,32)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([color_jitter],p=0.8),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=int(0.1*32)),
    transforms.ToTensor(),
])
#Augmentation 2
train_transform = DuplicatedCompose([
    transforms.RandomResizedCrop((32,32), scale=(0.2, 1.)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
'''
#Augmentation 1
train_transform = DuplicatedCompose([
    transforms.RandomResizedCrop((32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
])
from torch.utils.data import DataLoader

train_dataset = datasets.CIFAR10(root=dataset_path,
                                 train=True,
                                 download=True,
                                 transform=train_transform
                                )

train_loader = DataLoader(train_dataset,
                          batch_size=B,
                          num_workers=4,
                          shuffle=True,
                          drop_last=True
                         )

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

def loss_calc(x, y):
    return 2 - 2 * (F.normalize(x, dim=-1, p=2) * F.normalize(y, dim=-1, p=2)).sum(dim=-1)

class BYOL(torch.nn.Module):

    def __init__(self):
        super(BYOL, self).__init__()
        self.online_rep = ResNetCifar(depth=98, width=1, classes=10)
        self.target_rep = ResNetCifar(depth=98, width=1, classes=10)
        self.m = 0.99
        self.online_proj = MLP(64, 256, 4096)
        self.target_proj = MLP(64, 256, 4096)

        self.online_pred = MLP(256, 256, 4096)
        for pq, pk in zip(self.online_rep.parameters(), self.target_rep.parameters()):
            pk.data.copy_(pq.data) 
            pk.requires_grad = False  

    def forward(self, t1, t2):
        with torch.no_grad():
            for pq, pk in zip(self.online_rep.parameters(), self.target_rep.parameters()):
                pk.data = pk.data * self.m + pq.data * (1. - self.m)

        online_y_t1 = self.online_rep(t1)
        #print(t1.shape, online_y_t1.shape)
        online_z_t1 = self.online_proj(online_y_t1)
        online_q_t1 = self.online_pred(online_z_t1)

        online_y_t2 = self.online_rep(t2)
        online_z_t2 = self.online_proj(online_y_t2)
        online_q_t2 = self.online_pred(online_z_t2)

        with torch.no_grad():

            target_y_t1 = self.target_rep(t1)
            target_z_t1 = self.target_proj(target_y_t1)
            target_y_t2 = self.target_rep(t2)
            target_z_t2 = self.target_proj(target_y_t2)

        loss1 = loss_calc(online_q_t1, target_z_t2.detach())
        loss2 = loss_calc(online_q_t2, target_z_t1.detach())
        loss = loss1 + loss2
        return loss.sum(dim=-1)


def train(net, loader):
    optimizer = SGD_with_lars(net.parameters(), lr=0.1, momentum = 0.9, weight_decay = 1e-6)
    
    from warmup_scheduler import GradualWarmupScheduler
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=20, after_scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=180))
    
    train_start = time.time()
    
    for epoch in range(1, 100 + 1):
        print('hi')
        train_loss = 0
        net.train()
        epoch_start = time.time()
        for idx, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            
            dat1 = data[0].cuda()
            dat2 = data[1].cuda()
            loss = net(dat1, dat2)
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        train_loss /= (idx + 1)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        print("Epoch\t", epoch, 
              "\tLoss\t", train_loss, 
              "\tTime\t", epoch_time,
             )
        
    elapsed_train_time = time.time() - train_start
    print('Finished training. Train time was:', elapsed_train_time)

GPU_NUM = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NUM

net = BYOL()
net.cuda()
train(net, train_loader)
torch.save(net.state_dict(), pretrained_path)


net = BYOL()
net.load_state_dict(torch.load(pretrained_path))
net.eval()
net.cuda()
class BYOL_Classification(nn.Module):
    def __init__(self, net, num_classes=10):
        super(BYOL_Classification, self).__init__()
        
        self.num_classes = num_classes
        
        self.feat = net
        self.classifier = nn.Linear(64,num_classes) 
    
    def forward(self, x, norm_feat=False):
        feat = self.feat(x)
        logit = self.classifier(feat)
        return feat, logit
        
def train2(net, train_loader, test_loader):
    
    loss_fn = nn.CrossEntropyLoss()
    net2 = BYOL_Classification(net,10)
  
    net2.eval()
    net2.cuda()
    for pq in net.parameters():
        pq.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net2.parameters()), lr=1e-3)
    from warmup_scheduler import GradualWarmupScheduler
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=20, after_scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=80))
    
    train_start = time.time()
    for epoch in range(1, 100 + 1):
        
        train_loss = 0
        net2.train()
        
        epoch_start = time.time()
        for idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.cuda()
            target = target.cuda()
            data = net2(data)[1]
            loss = loss_fn(data, target)
            
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        train_loss /= (idx + 1)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        if epoch % 10 == 0:
          net.eval()
          total = 0.0
          correct = 0.0
          for test_data in test_loader:
            images, labels = test_data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net2(images)[1]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

          print("Epoch\t",epoch,"\tTest accuracy\t",correct/total*100)

    elapsed_train_time = time.time() - train_start
    print('Finished training. Train time was:', elapsed_train_time)

transform2 = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset2 = datasets.CIFAR10(root=dataset_path,
                                 train=True,
                                 download=True,
                                 transform=transform2
                                )

test_dataset2 = datasets.CIFAR10(root=dataset_path,
                                 train=False,
                                 download=True,
                                 transform=transform2
                                )

train_loader2 = DataLoader(train_dataset2,
                          batch_size=B,
                          num_workers=4,
                          shuffle=True,
                          drop_last=True
                         )

test_loader2 = DataLoader(test_dataset2,
                          batch_size=B,
                          num_workers=4,
                          shuffle=True,
                          drop_last=True
                         )

train2(net.online_rep, train_loader2, test_loader2)
