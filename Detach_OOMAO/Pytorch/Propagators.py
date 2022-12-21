import numpy as np
import math
from mpmath import *
import scipy.special as scp
import torch
import torchvision.transforms.functional as F
from oomao_functions import *
from torch import unsqueeze as UNZ

def CreateModulationPhasor(wfs):
    x = np.arange(0,wfs.fovInPixel,1)/wfs.fovInPixel
    vv, uu = np.meshgrid(x,x)
    r,o = cart2pol(uu,vv)
    nTheta = np.round(2*math.pi*wfs.samp*wfs.modulation)
    ModPhasor = np.zeros((wfs.fovInPixel,wfs.fovInPixel,np.int16(nTheta)),dtype=complex)

    for kTheta in range(np.int16(nTheta)):

        theta = (kTheta)*2*math.pi/nTheta
        ph = math.pi*4*wfs.modulation*wfs.samp*r*np.cos(o+theta)
        fftPhasor = np.exp(-1j*ph)
        ModPhasor[:,:,kTheta] = fftPhasor
        
    return(ModPhasor)


def AddPhotonNoise(y,wfs):
    buffer    = y + wfs.nPhotonBackground
    y = y + torch.normal(0,1,size=y.shape).cuda()*(y + wfs.nPhotonBackground)
    index     = y<0
    y[index] = buffer[index]
    y = wfs.quantumEfficiency*y
    return y

def Prop2VanillaPyrWFS_torch(phaseMap,wfs):
    
    nTheta = np.round(2*math.pi*wfs.samp*wfs.modulation)
    nTheta = torch.tensor(nTheta)
    PyrQ  = torch.zeros((wfs.fovInPixel,wfs.fovInPixel))
    pyrMask = torch.unsqueeze(torch.tensor(wfs.pyrMask),0)
    pupil = torch.tensor(wfs.pupil)        
    pyrPupil = pupil*torch.exp(1j*phaseMap)
    subscale = 1/(2*wfs.samp)
    sx = torch.round(torch.tensor(wfs.fovInPixel*subscale)).to(torch.int16)   
    npv = ((wfs.fovInPixel-sx)/2).to(torch.int16)
    PyrQ = torch.unsqueeze(torch.nn.functional.pad(pyrPupil,(npv,npv,npv,npv), "constant", 0),1)
    
    if nTheta > 0:
        I4Q4 =  torch.zeros((wfs.fovInPixel,wfs.fovInPixel))
        ModPhasor = torch.permute(wfs.ModPhasor,(2,0,1))
        buf = PyrQ*ModPhasor
        buf = torch.fft.fft2(buf)*pyrMask 
        buf = torch.fft.fft2(buf)      
        I4Q4 = torch.sum(torch.abs(buf)**2 ,1) 
        I4Q = I4Q4/nTheta
    else:
        buf = torch.fft.fft2(PyrQ)*pyrMask
        I4Q = torch.abs(torch.fft.fft2(buf))**2  
    I4Q = F.resize(I4Q,(sx,sx))*2*wfs.samp
    return(I4Q)



def Prop2OptimizePyrWFS_torch(phaseMap,DE,wfs):
    
    nTheta = np.round(2*math.pi*wfs.samp*wfs.modulation)
    nTheta = torch.tensor(nTheta)
    PyrQ  = torch.zeros((wfs.fovInPixel,wfs.fovInPixel))
    pupil = wfs.pupil  
    pyrMask = UNZ(UNZ(torch.tensor(wfs.pyrMask),0),0)  
    pyrPupil = pupil*torch.exp(1j*phaseMap)
    subscale = 1/(2*wfs.samp)
    sx = torch.round(wfs.fovInPixel*subscale).to(torch.int16)   
    npv = ((wfs.fovInPixel-sx)/2).to(torch.int16)
    PyrQ = torch.nn.functional.pad(pyrPupil,(npv,npv,npv,npv), "constant", 0)
    if nTheta > 0:
        I4Q4 =  torch.zeros((1,1,wfs.fovInPixel,wfs.fovInPixel))
        buf = PyrQ*wfs.ModPhasor
        buf = torch.fft.fft2(buf)*pyrMask*DE 
        buf = torch.fft.fft2(buf)      
        I4Q4 = torch.sum(torch.abs(buf)**2 ,1) 
        I4Q = I4Q4/nTheta
    else:
        buf = torch.fft.fft2(PyrQ)*DE
        I4Q = torch.abs(torch.fft.fft2(buf))**2
    I4Q = F.resize(I4Q,(sx,sx))*2*wfs.samp
    return(I4Q)


def Prop2OptimizeDWFS_torch(phaseMap,DE,wfs):
    
    nTheta = np.round(2*math.pi*wfs.samp*wfs.modulation)
    nTheta = torch.tensor(nTheta)
    PyrQ  = torch.zeros((wfs.fovInPixel,wfs.fovInPixel))
    pupil = wfs.pupil    
    pyrPupil = pupil*torch.exp(1j*phaseMap)
    subscale = 1/(2*wfs.samp)
    sx = torch.round(wfs.fovInPixel*subscale).to(torch.int16)   
    npv = ((wfs.fovInPixel-sx)/2).to(torch.int16)
    PyrQ = torch.nn.functional.pad(pyrPupil,(npv,npv,npv,npv), "constant", 0)
    if nTheta > 0:
        I4Q4 =  torch.zeros((wfs.fovInPixel,wfs.fovInPixel))
        ModPhasor = torch.permute(wfs.ModPhasor,(2,0,1))
        buf = PyrQ*ModPhasor
        buf = torch.fft.fft2(buf)*DE 
        buf = torch.fft.fft2(buf)      
        I4Q4 = torch.sum(torch.abs(buf)**2 ,2) 
        I4Q = I4Q4/nTheta
        I4Q = I4Q
    else:
        buf = torch.fft.fft2(PyrQ)*DE
        I4Q = torch.abs(torch.fft.fft2(buf))**2
    I4Q = F.resize(I4Q,(sx,sx))*2*wfs.samp
    return(I4Q)

