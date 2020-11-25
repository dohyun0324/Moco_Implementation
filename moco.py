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
import torchvision.models as models

class Downsample(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(Downsample, self).__init__()
        self.avg = nn.AvgPool2d(stride)
        assert nOut % nIn == 0
        self.expand_ratio = nOut // nIn

    def forward(self, x):
        x = self.avg(x)
        return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
    
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

img_size = (32, 32)

color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

train_transform = DuplicatedCompose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

from torch.utils.data import DataLoader

train_dataset = datasets.CIFAR10(root='../../../home_klimt/dohyun.kim/',
                                 train=True,
                                 download=True,
                                 transform=train_transform
                                )

train_loader = DataLoader(train_dataset,
                          batch_size=256,
                          num_workers=4,
                          shuffle=True,
                          drop_last=True
                         )


class NTXentLoss(torch.nn.Module):

    def __init__(self, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.cuda()

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
     #   print(l_pos)
       # print(r_pos)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits = logits / self.temperature
       # print(positives.shape)
     #   print(negatives.shape)
       # print(logits.shape)

        labels = torch.zeros(2 * self.batch_size).cuda().long()
#        print(labels)
        loss = self.criterion(logits, labels)
     #   print(loss)
        return loss / (2 * self.batch_size)

from torch.optim.optimizer import Optimizer, required

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

def train(net, loader):
    
    loss_fn = NTXentLoss(encoder = models.resnet50, batch_size=256, temperature=0.05, use_cosine_similarity=True)
    
    ### IMPLEMENTATION 4-2 ###
    ### 1. Use SGD_with_lars with
    ### lr = 0.1 * batch_size / 256
    ### momentum = 0.9
    ### weight_decay = 1e-6
    optimizer = SGD_with_lars(net.parameters(), lr=0.1, momentum = 0.9, weight_decay = 1e-6)
    
    from warmup_scheduler import GradualWarmupScheduler
    ### 2. Use GradualWarmupScheduler with
    ### multiplier = 1
    ### total_epoch = 1/10 of total epochs
    ### after_scheduler = optim.lr_scheduler.CosineAnnealingLR
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=20, after_scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=180))
    
    train_start = time.time()
    
    for epoch in range(1, 200 + 1):
        train_loss = 0
        net.train()
        epoch_start = time.time()
        for idx, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            
            ### 3. data variable contains two augmented images
            ### -1. send them to your GPU by calling .cuda()
            ### -2. forward each of them to net
            ### -3. compute the InfoNCE loss
            
            dat1 = data[0].cuda()
            dat2 = data[1].cuda()
            loss = loss_fn(dat1, dat2)
            ### IMPLEMENTATION ENDS HERE ###
            
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

net = SimCLRNet(26, 1, 10, 32)

net.cuda()
train(net, train_loader)
torch.save(net.state_dict(), '../../../home_klimt/dohyun.kim/pretrained.pt')

net = SimCLRNet(26, 1, 10, 32)
net.load_state_dict(torch.load('../../../home_klimt/dohyun.kim/pretrained.pt'))
net.eval()
net.cuda()
def train2(net, train_loader, test_loader):
    
    loss_fn = nn.CrossEntropyLoss()
  
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3)
    from warmup_scheduler import GradualWarmupScheduler
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=20, after_scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=80))
    
    train_start = time.time()
    
    for epoch in range(1, 100 + 1):
        
        train_loss = 0
        net.train()
        
        epoch_start = time.time()
        for idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.cuda()
            target = target.cuda()
            data = net(data)[2]
            loss = loss_fn(data, target)
            ### IMPLEMENTATION ENDS HERE ###
            
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        train_loss /= (idx + 1)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
       # print("Epoch\t", epoch, 
       #       "\tLoss\t", train_loss, 
       #       "\tTime\t", epoch_time,
       #      )
        
        if epoch % 10 == 0:
          net.eval()
          total = 0.0
          correct = 0.0
          for test_data in test_loader:
            images, labels = test_data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)[2]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

          print("Epoch\t",epoch,"\tTest accuracy\t",correct/total*100)

    elapsed_train_time = time.time() - train_start
    print('Finished training. Train time was:', elapsed_train_time)

transform2 = transforms.Compose([
    transforms.ToTensor(),
])
cnt=0
for p in net.feat.parameters():
    p.requires_grad = False
    cnt = cnt + 1
print(cnt)

train_dataset2 = datasets.CIFAR10(root='.',
                                 train=True,
                                 download=True,
                                 transform=transform2
                                )

test_dataset2 = datasets.CIFAR10(root='.',
                                 train=False,
                                 download=True,
                                 transform=transform2
                                )

train_loader2 = DataLoader(train_dataset2,
                          batch_size=256,
                          num_workers=4,
                          shuffle=True,
                          drop_last=True
                         )

test_loader2 = DataLoader(test_dataset2,
                          batch_size=256,
                          num_workers=4,
                          shuffle=True,
                          drop_last=True
                         )

train2(net, train_loader2, test_loader2)
