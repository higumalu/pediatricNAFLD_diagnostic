# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 11:29:35 2023

@author: higumalu
"""
import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models



class samsung_nafld(nn.Module):
    def __init__(self, num_class=5):
        super(samsung_nafld, self).__init__()
        self.b_en =  models.vgg16(pretrained=False)
        self.b_en.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.a_en =  models.vgg16(pretrained=False)
        self.a_en.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.m_en =  models.vgg16(pretrained=False)
        self.m_en.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.dense = nn.Sequential(
                        nn.Linear(3000, 1024),
                        nn.ReLU(True),
                        nn.Dropout(p=0.1),
                        nn.Linear(1024,num_class)
                        )
        
        
    def forward(self, imgs):
        
        bx = self.b_en(imgs[0])
        ax = self.a_en(imgs[1])
        mx = self.m_en(imgs[2])
        
        #bx = self.b_en(bx)
        #ax = self.b_en(ax)
        #mx = self.b_en(mx)
        x = torch.cat((bx,ax,mx),1)
        x = self.dense(x)
        
        return x
    
    
if __name__ == '__main__':
    
    model = samsung_nafld(num_class = 5)
    b = torch.randn(1, 1, 1024, 256)
    a = torch.randn(1, 1, 1024, 256)
    m = torch.randn(1, 1, 1024, 256)
    
    #imgs=torch.stack([b, a, m],dim=1)
    imgs=[b, a, m]
    preds = model(imgs)
    #print(model)
    print(preds)
    from thop import profile, clever_format
    macs, params = profile(model, inputs=(imgs, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
    
