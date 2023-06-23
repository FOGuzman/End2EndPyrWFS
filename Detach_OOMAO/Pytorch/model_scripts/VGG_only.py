import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
import numpy as np
from functions.oomao_functions import *
from functions.phaseGenerators import *
from functions.Propagators import *
from torch import unsqueeze as UNZ

class VGG(nn.Module):
    def __init__(self, num_classes=10,resolution=32):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * (resolution // 4) * (resolution // 4), 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class OptimizedPyramid(nn.Module):
    def __init__(self, wfs):
        super().__init__()
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
            self.ModPhasor = torch.permute(torch.tensor(wfs.ModPhasor),(2,0,1))
        self.fovInPixel    = torch.tensor(wfs.fovInPixel)
        self.pupil = torch.tensor(wfs.pupil)
        self.pyrMask = torch.tensor(wfs.pyrMask,dtype=torch.complex64)
        self.modes = torch.tensor(wfs.modes)       
        self.BatchModes = torch.zeros(size=(len(self.jModes),1,self.nPxPup,self.nPxPup)).cuda()
        for k in range(len(self.jModes)):           
            zim = torch.reshape(self.modes[:,k],(self.nPxPup,self.nPxPup))
            zim = torch.unsqueeze(zim,0).cuda()
            self.BatchModes[k,:,:,:] = zim

        self.pupilLogical = torch.tensor(wfs.pupilLogical)
        self.Flat = torch.ones((self.nPxPup,self.nPxPup))*self.pupilLogical
        self.Flat = UNZ(UNZ(self.Flat,0),0).cuda()

        self.NN =  VGG(num_classes=len(self.jModes),resolution=self.nPxPup)

        self.NN = self.NN.cuda()
        OL1 = torch.ones((wfs.fovInPixel,wfs.fovInPixel))
        self.OL1  = nn.Parameter(OL1)
        
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
        #propagation of X
        Ip = Prop2VanillaPyrWFS_torch(inputs,self)
        #Photon noise
        if self.PhotonNoise == 1:
            Ip = AddPhotonNoise(Ip,self)          
        #Read out noise 
        if self.ReadoutNoise != 0:
            Ip = Ip + torch.normal(0,self.ReadoutNoise,size=Ip.shape).cuda()   
        
        # Normalization
        Inorm = torch.sum(torch.sum(torch.sum(Ip,-1),-1),-1)
        Ip = Ip/UNZ(UNZ(UNZ(Inorm,-1),-1),-1)
        # Estimation
        y = self.NN(Ip).permute(1,0)
        return y
    
       
       
            
class PyrModel(nn.Module):
    def __init__(self,wfs):
        super(PyrModel,self).__init__()
        self.prop = OptimizedPyramid(wfs)

    def forward(self, x):
        return self.prop(x)   