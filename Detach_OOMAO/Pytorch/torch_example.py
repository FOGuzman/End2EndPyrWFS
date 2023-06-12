#%%
from loadData import Imgdataset
import torch.optim as optim
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
from functions.oomao_functions import *
from functions.phaseGenerators import *
from functions.customLoss      import RMSE
from functions.utils import *
from math import sqrt, pi
from matplotlib import pyplot as plt



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
n_gpu = torch.cuda.device_count()
print(torch.cuda.is_available())
print('The number of GPU is {}'.format(n_gpu))




parser = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')

parser.add_argument('--modulation', default=0, type=int, help='Pyramid modulation')
parser.add_argument('--samp', default=2, type=int, help='Sampling')
parser.add_argument('--D', default=8, type=int, help='Telescope Diameter [m]')
parser.add_argument('--nPxPup', default=128, type=int, help='Pupil Resolution')
parser.add_argument('--rooftop', default=[0,0], type=float)
parser.add_argument('--alpha', default=pi/2, type=float)
parser.add_argument('--zModes', default=[2,128], type=int, help='Reconstruction Zernikes')
parser.add_argument('--PupilConstrain', default=0, type=int, help='Limit information only on pupils of PyrWFS')
parser.add_argument('--ReadoutNoise', default=0, type=float)
parser.add_argument('--PhotonNoise', default=0, type=float)
parser.add_argument('--nPhotonBackground', default=0, type=float)
parser.add_argument('--quantumEfficiency', default=1, type=float)
wfs = parser.parse_args()

wfs.fovInPixel    = wfs.nPxPup*2*wfs.samp 
wfs.pupil = CreateTelescopePupil(wfs.nPxPup,"disc")
wfs.pyrMask = createPyrMask(wfs)
wfs.jModes = torch.arange(wfs.zModes[0], wfs.zModes[1]+1)
wfs.pupilLogical = wfs.pupil!=0
wfs.modes = CreateZernikePolynomials(wfs)
wfs.amplitude = 0.2 #small for low noise systems
wfs.ModPhasor = torch.tensor(CreateModulationPhasor(wfs))

###### Flat prop
zim = np.reshape(wfs.modes[:,0],(wfs.nPxPup,wfs.nPxPup))
zim = np.ones((wfs.nPxPup,wfs.nPxPup))*wfs.pupilLogical
zim = torch.tensor(np.expand_dims(zim,0))
I_0 = Propagate2Pyramid_torch(zim,wfs)
I_0 = I_0/torch.sum(I_0)
###### Calibration
IM = None
gain = 0.1
for k in range(len(wfs.jModes)):           
            zim = torch.reshape(torch.tensor(wfs.modes[:,k]),(wfs.nPxPup,wfs.nPxPup))
            zim = torch.unsqueeze(zim,0)
            #push
            z = gain*zim
            I4Q = Propagate2Pyramid_torch(z,wfs)
            sp = I4Q/torch.sum(I4Q)-I_0
            
            #pull
            I4Q = Propagate2Pyramid_torch(-z,wfs)
            sm = I4Q/torch.sum(I4Q)-I_0
            
            MZc = 0.5*(sp-sm)/gain
            Zv = torch.unsqueeze(torch.reshape(MZc,[-1]),1)
            if IM is not None:
                IM = torch.concat([IM,Zv],1)
            else:
                IM = Zv

CM = torch.linalg.pinv(IM)



#%% #####Estimation
phaseMap,Zgt = ZernikeProfile(3,0.5,wfs)
phaseMap = torch.tensor(phaseMap)
Ip = Propagate2Pyramid_torch(phaseMap,wfs)
Inorm = torch.sum(torch.sum(Ip,-1),-1)
Inorm = torch.unsqueeze(torch.unsqueeze(Inorm,-1),-1)
Ip = Ip/Inorm

Zpyr = torch.matmul(CM,torch.transpose(torch.reshape(Ip,[3,-1]),0,1))


plot_tensor(phaseMap[0,:,:])
plot_tensor(Ip[0,:,:])

fig, ax = plt.subplots()
plt.title("Estimation")
line1, = ax.plot(Zgt[:,0])
line2, = ax.plot(Zpyr[:,0])
plt.show(block=False)



#% Control matrix    
    
phCM = torch.linalg.pinv(torch.tensor(wfs.modes))
#%% ############# Fourier Phase

from phaseGenerators import *

nLenslet      = 16                 # plens res
resAO         = 2*nLenslet+1       # AO resolution
r0            = 0.11           
L0            = 25
fR0           = 1.2
noiseVariance = 0.7
n_lvl         = 0.1
nTimes        = wfs.fovInPixel/resAO



t0 = time.time()
atm = GetTurbulenceParameters(wfs,resAO,nLenslet,r0,L0,fR0,noiseVariance,nTimes,n_lvl)
phaseMap,Zgt = GetPhaseMapAndZernike(atm,phCM)
t1 = time.time()

time1 = t1-t0

#Test CUDA
if torch.cuda.is_available() == 1:
    t0 = time.time()
    psdAO_mean = torch.tensor(atm['psdAO_mean'])
    N = torch.tensor(atm['N'],dtype=torch.float64)
    fourierSampling = torch.tensor(atm['fourierSampling'])
    idx = torch.tensor(atm['idx'])
    pupil = torch.tensor(atm['pupil'])
    nPxPup = torch.tensor(atm['nPxPup'])
    phaseMap,Zgt = GetPhaseMapAndZernike_CUDA(psdAO_mean.cuda(),N.cuda(),fourierSampling.cuda(),idx.cuda(),pupil.cuda(),nPxPup.cuda(),phCM.cuda())
    phaseMap = phaseMap.cpu()
    Zgt = Zgt.cpu()
    t1 = time.time()
time2 = t1-t0


phaseMap = torch.unsqueeze(phaseMap,0)
Ip = Prop2VanillaPyrWFS_torch(phaseMap,wfs)
Ip = Ip/torch.sum(Ip)

Zpyr = torch.matmul(CM,torch.reshape(Ip,[-1]))


plot_tensor(phaseMap)
plot_tensor(Ip)

fig, ax = plt.subplots()
plt.title("Estimation")
line1, = ax.plot(Zgt)
line2, = ax.plot(Zpyr)
plt.show()