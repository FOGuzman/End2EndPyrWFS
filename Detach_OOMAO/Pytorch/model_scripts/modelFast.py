import torch.nn as nn
import torch
import numpy as np
from functions.oomao_functions import *
from functions.phaseGenerators import *
from functions.Propagators import *
from torch import unsqueeze as UNZ

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
            self.ModPhasor = torch.permute(torch.tensor(wfs.ModPhasor),(2,0,1))
        self.fovInPixel    = torch.tensor(wfs.fovInPixel)
        self.pupil = torch.tensor(wfs.pupil)
        self.pyrMask = torch.tensor(wfs.pyrMask)
        self.modes = torch.tensor(wfs.modes)       
        self.BatchModes = torch.zeros(size=(len(self.jModes),1,self.nPxPup,self.nPxPup)).cuda()
        for k in range(len(self.jModes)):           
            zim = torch.reshape(self.modes[:,k],(self.nPxPup,self.nPxPup))
            zim = torch.unsqueeze(zim,0).cuda()
            self.BatchModes[k,:,:,:] = zim

        self.pupilLogical = torch.tensor(wfs.pupilLogical)
        self.Flat = torch.ones((self.nPxPup,self.nPxPup))*self.pupilLogical
        self.Flat = UNZ(UNZ(self.Flat,0),0).cuda()
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
        OL1 = UNZ(UNZ(torch.exp(1j * self.OL1),0),0)       
        # Flat prop
        I_0 = Prop2OptimizePyrWFS_torch(self.Flat,OL1,self)
        self.I_0 = I_0/torch.sum(I_0)
        
        gain = 0.1
        
        # Calibration matrix as batch
        z = self.BatchModes*gain
        I4Q = Prop2OptimizePyrWFS_torch(z,OL1,self)
        spnorm = UNZ(UNZ(UNZ(torch.sum(torch.sum(torch.sum(I4Q,dim=-1),dim=-1),dim=-1),-1),-1),-1)
        sp = I4Q/spnorm-self.I_0

        I4Q = Prop2OptimizePyrWFS_torch(-z,OL1,self)
        smnorm = UNZ(UNZ(UNZ(torch.sum(torch.sum(torch.sum(I4Q,dim=-1),dim=-1),dim=-1),-1),-1),-1)
        sm = I4Q/smnorm-self.I_0

        MZc = 0.5*(sp-sm)/gain

        MZc = MZc.view(MZc.shape[0],self.nPxPup**2)
        MZc = MZc.permute(1,0)


        CM = torch.linalg.pinv(MZc)       
        #propagation of X
        Ip = Prop2OptimizePyrWFS_torch(inputs,OL1,self)
        #Photon noise
        if self.PhotonNoise == 1:
            Ip = AddPhotonNoise(Ip,self)          
        #Read out noise 
        if self.ReadoutNoise != 0:
            Ip = Ip + torch.normal(0,self.ReadoutNoise,size=Ip.shape).cuda()   
        
        # Normalization
        Inorm = torch.sum(torch.sum(torch.sum(Ip,-1),-1),-1)
        Ip = Ip/UNZ(UNZ(Inorm,-1),-1)-self.I_0
        # Estimation
        y = torch.matmul(CM,torch.transpose(torch.reshape(Ip,[Ip.shape[0],-1]),0,1))
        return y
    
    
class PhaseConstraint(object):
    def __init__(self):
        pass
    
    def __call__(self,module):
        if hasattr(module,'OL1'):
            w=module.OL1.data
            w=w.clamp(-8*math.pi,8*math.pi)
            module.OL1.data=w    
       
            
class PyrModel(nn.Module):
    def __init__(self,wfs):
        super(PyrModel,self).__init__()
        self.prop = OptimizedPyramid(wfs)

    def forward(self, x):
        return self.prop(x)   