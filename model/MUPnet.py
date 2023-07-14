# -*- coding: utf-8 -*-
"""
Created on 2022/07/13

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
            
        self.dwconv = nn.Conv2d(in_c, mid_c, 
                                kernel_size=ks,  stride=1, 
                                padding=(int(ks[0]/2),int(ks[1]/2)), bias=False)
        #self.norm = nn.LayerNorm(mid_c, eps = 1e-6)
        self.norm = nn.GroupNorm(mid_c, mid_c)
        
        self.dpconv = nn.Conv2d(mid_c, out_c, 
                                kernel_size=(1,1),  stride=1, 
                                padding=0, bias=False)
        self.act = nn.GELU()
        
    def forward(self, x):
        x_in = x
        x = self.dwconv(x)
        #x = x.permute(0, 2, 3, 1) #Layernorm used
        x = self.norm(x)
        #x = x.permute(0, 3, 1, 2)  #Layernorm used
        x = self.act(x)
        x = self.dpconv(x)
        
        return x
    
    
class DOWN(nn.Module):
    def __init__(self, in_c, out_c, ks=(9,3), mid_c=None):
        super().__init__()
        if not mid_c:
            mid_c = out_c
            
        self.downblock = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=(2, 2), stride=2, padding=1, bias=False),
            dual(in_c, out_c, ks, mid_c)
        )
        
    def forward(self, x):
        return self.downblock(x)
        
class dualv2(nn.Module):
    def __init__(self, in_c, out_c, ks=(9,3), mid_c=None):
        super().__init__()
        if not mid_c:
            mid_c = out_c
            
        self.dwconv = nn.Conv2d(in_c, mid_c, 
                                kernel_size=ks,  stride=1, 
                                padding=(int(ks[0]/2),int(ks[1]/2)), bias=False)
        #self.norm = nn.LayerNorm(mid_c, eps = 1e-6)
        self.norm = nn.GroupNorm(mid_c, mid_c)
        
        self.dpconv = nn.Conv2d(mid_c, out_c, 
                                kernel_size=(1,1),  stride=1, 
                                padding=0, bias=False)
        self.act = nn.GELU()
        
    def forward(self, x):
        x_in = x
        x = self.dwconv(x)
        #x = x.permute(0, 2, 3, 1) #Layernorm used
        x = self.norm(x)
        #x = x.permute(0, 3, 1, 2)  #Layernorm used
        x = self.act(x)
        x = self.dpconv(x)
        
        return x

class MUPv2(nn.Module):
    def __init__(self):
        super(MUPv2, self).__init__()
        #input size = 1024,256 
        self.head = dual(7, 64, (9,7), mid_c=64)    #in = 1 or 6 or 7
        self.down1 = DOWN(64, 128, (9,7), mid_c=128)   #512,128
        self.down2 = DOWN(128, 128, (9,7), mid_c=256)   #256,64
        #cat
        self.down3 = DOWN(128, 128, (9,7), mid_c=256)   #128,32
        self.down4 = DOWN(128, 256, (9,7), mid_c=256)   #64,16
        self.down5 = DOWN(256, 256, (9,7), mid_c=512)   #32,8
        self.down6 = DOWN(256, 256, (9,7), mid_c=512)   #16,4
        
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(21760,512),
            nn.Linear(512,256),
            nn.Dropout(0.1),
            nn.Linear(256,5), #5 or 2
            #nn.LogSoftmax(dim=1)
            )
    def forward(self, x):
        x = self.head(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.down6(x)
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
        
      