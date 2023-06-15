from functions.loadData import Imgdataset
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
from functions.phaseGeneratorsCuda import *
from functions.customLoss      import RMSE
from functions.utils import *
from functions.Propagators import *
from math import sqrt, pi
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
matplotlib.interactive(True)


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
n_gpu = torch.cuda.device_count()
print(torch.cuda.is_available())
print('The number of GPU is {}'.format(n_gpu))




parser = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')

parser.add_argument('--modulation', default=0, type=int, help='Pyramid modulation')
parser.add_argument('--samp', default=2, type=int, help='Sampling')
parser.add_argument('--D', default=8, type=int, help='Telescope Diameter [m]')
parser.add_argument('--nPxPup', default=224, type=int, help='Pupil Resolution')
parser.add_argument('--rooftop', default=[0,0], type=float)
parser.add_argument('--alpha', default=pi/2, type=float)
parser.add_argument('--zModes', default=[2,36], type=int, help='Reconstruction Zernikes')
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

##TO CUDA
wfs.pupil = torch.from_numpy(wfs.pupil).cuda().float()
wfs.pyrMask = torch.from_numpy(wfs.pyrMask).cuda().cfloat()
wfs.fovInPixel    = torch.tensor(wfs.fovInPixel).cuda()
wfs.modes = torch.tensor(wfs.modes).cuda().float()
wfs.D     = torch.tensor(wfs.D).cuda()
IM_batch = torch.zeros(size=(len(wfs.jModes),1,wfs.nPxPup,wfs.nPxPup)).cuda()
for k in range(len(wfs.jModes)):           
    zim = torch.reshape(wfs.modes[:,k],(wfs.nPxPup,wfs.nPxPup))
    zim = torch.unsqueeze(zim,0).cuda()
    IM_batch[k,:,:,:] = zim



Flat = torch.ones((wfs.nPxPup,wfs.nPxPup))*wfs.pupilLogical
Flat = UNZ(UNZ(Flat,0),0).cuda()


###### Flat prop      
        # Flat prop
I_0 = Prop2VanillaPyrWFS_torch(Flat,wfs)
I_0 = I_0/torch.sum(I_0)

gain = 0.1

# Calibration matrix as batch
z = IM_batch*gain
I4Q = Prop2VanillaPyrWFS_torch(z,wfs)
spnorm = UNZ(UNZ(UNZ(torch.sum(torch.sum(torch.sum(I4Q,dim=-1),dim=-1),dim=-1),-1),-1),-1)
sp = I4Q/spnorm

I4Q = Prop2VanillaPyrWFS_torch(-z,wfs)
smnorm = UNZ(UNZ(UNZ(torch.sum(torch.sum(torch.sum(I4Q,dim=-1),dim=-1),dim=-1),-1),-1),-1)
sm = I4Q/smnorm

MZc = 0.5*(sp-sm)/gain

MZc = MZc.view(MZc.shape[0],wfs.nPxPup**2)
MZc = MZc.permute(1,0)

#% Control matrix
CM = torch.linalg.pinv(MZc)       
phCM = torch.linalg.pinv(torch.tensor(wfs.modes))
    
    

#%% ############# Fourier Phase

nLenslet      = torch.tensor(16).cuda()                 # plens res
resAO         = 2*nLenslet+1       # AO resolution
r0            = torch.tensor(0.11).cuda()            
L0            = torch.tensor(25).cuda()   
fR0           = torch.tensor(1.2).cuda()   
noiseVariance = torch.tensor(0.7).cuda()   
n_lvl         = torch.tensor(0.1).cuda()   
nTimes        = wfs.fovInPixel/resAO
batch_size    = 5


t0 = time.time()
atm = GetTurbulenceParameters(wfs,resAO,nLenslet,r0,L0,fR0,noiseVariance,nTimes,n_lvl)
phaseMap,Zgt = GetPhaseMapAndZernike(atm,phCM,batch_size)

Ip = Prop2VanillaPyrWFS_torch(phaseMap,wfs)
Inorm = torch.sum(torch.sum(torch.sum(Ip,-1),-1),-1)
Ip = Ip/UNZ(UNZ(UNZ(Inorm,-1),-1),-1)-I_0

Zpyr = torch.matmul(CM,torch.transpose(torch.reshape(Ip,[Ip.shape[0],-1]),0,1))


t1 = time.time()
time1 = t1-t0
plot_tensor(phaseMap[0,0,:,:].cpu())
plot_tensor(Ip[0,0,:,:].cpu())

fig, ax = plt.subplots()
plt.title("Estimation"+str(time1))
line1, = ax.plot(Zgt[:,0].cpu())
line2, = ax.plot(Zpyr[:,0].cpu())
plt.show()