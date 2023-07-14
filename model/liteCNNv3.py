# -*- coding: utf-8 -*-
"""
Created on 2022/06/11

@author: higumalu
"""
import torch.nn.functional as F
import torch.nn as nn
import torch

class dual(nn.Module):
    def __init__(self, in_c, out_c, ks=(9,3), mid_c=None):
        super().__init__()
        if not mid_c:
            mid_c = out_c
            
        self.conv = nn.Conv2d(in_c, mid_c, kernel_size=ks,  stride=1, padding=(int(ks[0]/2),int(ks[1]/2)), bias=False)
        self.norm = nn.LayerNorm(mid_c, eps = 1e-6)
        self.dwconv = nn.Conv2d(mid_c, out_c, kernel_size=(1,1),  stride=1, padding=0, bias=False)
        self.act = nn.GELU()
        
    def forward(self, x):
        x_in = x
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = self.act(x)
        return x
    
    
class DOWN(nn.Module):
    def __init__(self, in_c, out_c, ks=(9,3), mid_c=None):
        super().__init__()
        if not mid_c:
            mid_c = out_c
            
        self.downblock = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=(2, 2), stride=2, padding=1, bias=False),
            dual(in_c, out_c, ks=(7,7), mid_c=None)
        )
        
    def forward(self, x):
        return self.downblock(x)
        
        
    
class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
        
        self.conv1_1 = dual(3,4,4,(3,3))
        self.conv1_2 = dual(3,4,4,(5,5))
        self.conv2_1 = dual(8,8,8,(3,3))
        self.conv2_2 = dual(8,8,8,(5,5))
        self.conv3 = dual(16,32,16,(5,5))
        self.conv4 = dual(16,32,16,(3,3))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(27984,256),
            nn.Linear(256,128),
            #nn.Dropout(0.1),
            nn.Linear(128,10),
            nn.LogSoftmax(dim=1)
            )
    def forward(self, x):
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x = torch.cat([x1, x2], dim=1)
        
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x = torch.cat([x1, x2], dim=1)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        #print(x.size(1))
        x = self.fc(x)
        
        return x
        
        
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
      