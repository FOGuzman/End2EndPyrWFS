import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
import argparse
import random
from torch.autograd import Variable
from tqdm import tqdm
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as compare_ssim
from oomao_functions import *
from phaseGenerators import *
from math import sqrt, pi

class OptimizedPyramid(nn.Module):
    def __init__(self, wfs):
        super().__init__()
        self.batchSize = wfs.batchSize
        self.nPxPup = wfs.nPxPup
        self.modulation = wfs.modulation
        self.samp = wfs.samp
        self.rooftop = wfs.rooftop
        self.alpha = wfs.alpha
        self.jModes = wfs.jModes
        self.amplitude = wfs.amplitude
        self.ReadoutNoise = wfs.ReadoutNoise
        self.PhotonNoise = torch.tensor(wfs.PhotonNoise)
        self.quantumEfficiency = torch.tensor(wfs.quantumEfficiency)
        self.nPhotonBackground = torch.tensor(wfs.nPhotonBackground)
        if wfs.modulation > 0:
            self.ModPhasor = torch.tensor(wfs.ModPhasor)
        self.fovInPixel    = torch.tensor(wfs.fovInPixel)
        self.pupil = torch.tensor(wfs.pupil)
        self.pyrMask = torch.tensor(wfs.pyrMask)
        self.modes = torch.tensor(wfs.modes)
        self.pupilLogical = torch.tensor(wfs.pupilLogical)
        OL1 = torch.ones((wfs.fovInPixel,wfs.fovInPixel))
        OL1 = torch.tensor(np.angle(wfs.pyrMask))
        self.OL1  = nn.Parameter(OL1)

        # init weights
        nn.init.constant_(self.OL1,1)
        
        ## CUDA
        if torch.cuda.is_available() == 1:
            self.pyrMask = self.pyrMask.cuda()
            self.pupil = self.pupil.cuda()
            if wfs.modulation > 0:
                self.ModPhasor = self.ModPhasor.cuda()
        self.PhotonNoise = self.PhotonNoise.cuda()
        self.quantumEfficiency = self.quantumEfficiency.cuda()
        self.nPhotonBackground = self.nPhotonBackground.cuda()    
            
            

    def forward(self, inputs):
        OL1 = torch.exp(1j * self.OL1)       
        # Flat prop
        zim = torch.reshape(self.modes[:,0],(self.nPxPup,self.nPxPup))
        zim = torch.ones((self.nPxPup,self.nPxPup))*self.pupilLogical
        zim = torch.unsqueeze(zim,0).cuda()
        I_0 = Pro2OptPyrNoMod_torch(zim,OL1,self)
        self.I_0 = I_0/torch.sum(I_0)        
        IM = None
        gain = 0.1
        for k in range(len(self.jModes)):           
            zim = torch.reshape(self.modes[:,k],(self.nPxPup,self.nPxPup))
            zim = torch.unsqueeze(zim,0).cuda()
            #push
            z = gain*zim
            I4Q = Pro2OptPyrNoMod_torch(z,OL1,self)
            sp = I4Q/torch.sum(I4Q)-self.I_0
            
            #pull
            I4Q = Pro2OptPyrNoMod_torch(-z,OL1,self)
            sm = I4Q/torch.sum(I4Q)-self.I_0
            
            MZc = 0.5*(sp-sm)/gain
            Zv = torch.unsqueeze(torch.reshape(MZc,[-1]),1)
            if IM is not None:
                IM = torch.concat([IM,Zv],1)
            else:
                IM = Zv

        CM = torch.linalg.pinv(IM)       
        #propagation of X
        Ip = Pro2OptPyrNoMod_torch(inputs,OL1,self)
        #Photon noise
        if self.PhotonNoise == 1:
            Ip = AddPhotonNoise(Ip,self)          
        #Read out noise 
        if self.ReadoutNoise != 0:
            Ip = Ip + torch.normal(0,self.ReadoutNoise,size=Ip.shape).cuda()   
        
        # Normalization
        Inorm = torch.sum(torch.sum(Ip,-1),-1)
        Ip = Ip/torch.unsqueeze(torch.unsqueeze(Inorm,-1),-1)-self.I_0
        # Estimation
        y = torch.matmul(CM,torch.transpose(torch.reshape(Ip,[Ip.shape[0],-1]),0,1))
        return y
    
    
class PhaseConstraint(object):
    def __init__(self):
        pass
    
    def __call__(self,module):
        if hasattr(module,'OL1'):
            w=module.OL1.data
            w=w.clamp(-6*math.pi,6*math.pi)
            module.OL1.data=w    
       
            
class PyrModel(nn.Module):
    def __init__(self,wfs):
        super(PyrModel,self).__init__()
        self.prop = OptimizedPyramid(wfs)

    def forward(self, x):
        return self.prop(x)           
            