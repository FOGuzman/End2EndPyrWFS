from torch.utils.data import DataLoader
import importlib
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
import argparse
from functions.oomao_functions import *
from functions.phaseGenerators import *
from functions.customLoss      import RMSE
from functions.utils import *
from functions.Propagators import *
import random
from types import SimpleNamespace
import matplotlib.pyplot as plt


date = datetime.date.today()  





parser = argparse.ArgumentParser(description='Settings, Training and Pyramid Wavefron Sensor parameters')


parser.add_argument('--modulation', default=0, type=int, help='Pyramid modulation')
parser.add_argument('--samp', default=2, type=int, help='Over sampling for fourier')
parser.add_argument('--D', default=8, type=int, help='Telescope Diameter [m]')
parser.add_argument('--nPxPup', default=224, type=int, help='Pupil Resolution')
parser.add_argument('--rooftop', default=[0,0], type=eval,help='Pyramid rooftop (as in OOMAO)')
parser.add_argument('--alpha', default=np.pi/2, type=float,help='Pyramid angle (as in OOMAO)')
parser.add_argument('--zModes', default=[2,16], type=eval, help='Reconstruction Zernikes')
parser.add_argument('--ZernikeUnits', default=88, type=float,help='Zernike units (1 for normalized)')
parser.add_argument('--ReadoutNoise', default=0, type=float)
parser.add_argument('--PhotonNoise', default=0, type=float)
parser.add_argument('--nPhotonBackground', default=0, type=float)
parser.add_argument('--quantumEfficiency', default=1, type=float)

parser.add_argument('--D_r0', default=[90,10], type=eval, help='Range of r0 to create')
parser.add_argument('--datapoints', default=10, type=int, help='r0 intervals')
parser.add_argument('--dperR0', default=100, type=int, help='test per datapoint')

parser.add_argument('--models', nargs='+',default=['GCViT_only','modelFastplusGCViT'])
parser.add_argument('--checkpoints', nargs='+',default=
                    ['./model/nocap/GCViT/S2_R224_Z2-16_D8/checkpoint/PyrNet_epoch_34.pth',
                     './model/nocap/DE+GCViT/S2_R224_Z2-16_D8/checkpoint/PyrNet_epoch_358.pth'])


# Precalculations
wfs = parser.parse_args()
wfs.fovInPixel    = wfs.nPxPup*2*wfs.samp 
wfs.pupil = CreateTelescopePupil(wfs.nPxPup,"disc")
wfs.pyrMask = createPyrMask(wfs)
wfs.jModes = torch.arange(wfs.zModes[0], wfs.zModes[1]+1)
wfs.pupilLogical = wfs.pupil!=0
wfs.modes = CreateZernikePolynomials(wfs)
wfs.amplitude = 0.2 #small for low noise systems
wfs.ModPhasor = CreateModulationPhasor(wfs)


def clone_to_cuda(x):
    out = x
    if not isinstance(x, torch.Tensor):
        out = torch.tensor(out)  # Convert to a tensor
    if torch.cuda.is_available() and not out.is_cuda:
        out = out.cuda()   
    return out


wfs_cuda = SimpleNamespace()
wfs_cuda.pupil = clone_to_cuda(wfs.pupil)
wfs_cuda.fovInPixel = clone_to_cuda(wfs.fovInPixel)
wfs_cuda.pyrMask = clone_to_cuda(wfs.pyrMask)
wfs_cuda.samp = wfs.samp
wfs_cuda.modulation = wfs.modulation
model =[]
for k in range(len(wfs.models)):
    method = importlib.import_module("model_scripts."+wfs.models[k])
    single_model = method.PyrModel(wfs).cuda() 
    checkpoint = torch.load(wfs.checkpoints[k])
    single_model.load_state_dict(checkpoint.state_dict())
    single_model.eval()
    model.append(single_model)




nLenslet      = 16                 # plens res
resAO         = 2*nLenslet+1       # AO resolution           
L0            = 25
fR0           = 1
noiseVariance = 0.7
n_lvl         = 0.1
D             = wfs.D
nTimes        = wfs.fovInPixel/resAO


Dr0Vector = np.linspace(wfs.D_r0[0],wfs.D_r0[1],wfs.datapoints)
r0Vector = D/Dr0Vector

IM = None
    #%% Control matrix
for k in range(len(wfs.jModes)):
    imMode = torch.reshape(torch.tensor(wfs.modes[:,k]),(wfs.nPxPup,wfs.nPxPup))
    Zv = torch.unsqueeze(torch.reshape(imMode,[-1]),1)
    if IM is not None:
        IM = torch.concat([IM,Zv],1)
    else:
        IM = Zv

CMPhase = torch.linalg.pinv(IM).cuda()



###### Flat prop
zim = np.reshape(wfs.modes[:,0],(wfs.nPxPup,wfs.nPxPup))
zim = np.ones((wfs.nPxPup,wfs.nPxPup))*wfs.pupilLogical
zim = torch.tensor(np.expand_dims(zim,0))
I_0 = Prop2VanillaPyrWFS_torch(zim.cuda().float(),wfs_cuda)
I_0 = I_0/torch.sum(I_0)
###### Calibration
IM = None
gain = 0.1
for k in range(len(wfs.jModes)):           
            zim = torch.reshape(torch.tensor(wfs.modes[:,k]),(wfs.nPxPup,wfs.nPxPup))
            zim = torch.unsqueeze(zim,0)
            #push
            z = gain*zim.cuda().float()
            I4Q = Prop2VanillaPyrWFS_torch(z,wfs_cuda)
            sp = I4Q/torch.sum(I4Q)-I_0
            
            #pull
            I4Q = Prop2VanillaPyrWFS_torch(-z,wfs_cuda)
            sm = I4Q/torch.sum(I4Q)-I_0
            
            MZc = 0.5*(sp-sm)/gain
            Zv = torch.unsqueeze(torch.reshape(MZc,[-1]),1)
            if IM is not None:
                IM = torch.concat([IM,Zv],1)
            else:
                IM = Zv

CM = torch.linalg.pinv(IM)


def compute_rmse(vector1, vector2):
    mse = torch.mean((vector1 - vector2) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.detach().cpu().numpy()


ZFull = []
for k in range(len(wfs.models)+1):
    vsize = np.zeros((wfs.dperR0,wfs.datapoints))
    ZFull.append(vsize) 



for r0idx in range(wfs.datapoints):
    r0el = r0Vector[r0idx]
    for nridx in range(wfs.dperR0):

        atm = GetTurbulenceParameters(wfs,resAO,nLenslet,r0el,L0,fR0,noiseVariance,nTimes,n_lvl)
        psdAO_mean = torch.tensor(atm['psdAO_mean'])
        N = torch.tensor(atm['N'],dtype=torch.float64)
        fourierSampling = torch.tensor(atm['fourierSampling'])
        idx = torch.tensor(atm['idx'])
        pupil = torch.tensor(atm['pupil'])
        nPxPup = torch.tensor(atm['nPxPup'])
        phaseMap,Zgt = GetPhaseMapAndZernike_CUDA(psdAO_mean.cuda(),N.cuda(),fourierSampling.cuda(),idx.cuda(),pupil.cuda(),nPxPup.cuda(),CMPhase)
        phaseMap = phaseMap.float()
        Zgt = torch.unsqueeze(Zgt.float(),1)


        Ip = Prop2VanillaPyrWFS_torch(phaseMap,wfs_cuda)
        Ip = Ip/torch.sum(Ip)

        Zpyr = torch.matmul(CM,torch.reshape(Ip,[-1]))
        ZFull[0][nridx,r0idx] = compute_rmse(Zgt,Zpyr)
        Zest = []
        for m in range(len(wfs.models)):
            Z_single = model[m](torch.unsqueeze(torch.unsqueeze(phaseMap,0),0)).detach()
            ZFull[m+1][nridx,r0idx] = compute_rmse(Zgt,Z_single)



meanZ = []
stdZ  = []
meanZ.append(np.mean(ZFull[0],axis=0))     
meanZ.append(np.mean(ZFull[1],axis=0)) 
meanZ.append(np.mean(ZFull[2],axis=0)) 
stdZ.append(np.std(ZFull[0],axis=0))     
stdZ.append(np.std(ZFull[1],axis=0)) 
stdZ.append(np.std(ZFull[2],axis=0)) 




fig = plt.figure()


plt.errorbar(Dr0Vector, meanZ[0], yerr=stdZ[0], label='PyrWFS')
plt.errorbar(Dr0Vector, meanZ[1], yerr=stdZ[1], label='GCViT')
plt.errorbar(Dr0Vector, meanZ[2], yerr=stdZ[2], label='DE+GCViT')
plt.gca().invert_xaxis()        
plt.legend()
plt.show()        