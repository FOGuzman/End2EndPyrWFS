import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
import argparse
import random
from torch.distributions.uniform import Uniform as urand
from oomao_functions import *
from phaseGenerators import *
from math import sqrt, pi



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
n_gpu = torch.cuda.device_count()
print(torch.cuda.is_available())
print('The number of GPU is {}'.format(n_gpu))

tData = 2000
vData = 200
Zs = [-5,5]
main_fold = "./dataset_tilt/"



parser = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')

#parser.add_argument('--modulation', default=1, type=int, help='Pyramid modulation')
parser.add_argument('--samp', default=2, type=int, help='Sampling')
parser.add_argument('--D', default=8, type=int, help='Telescope Diameter [m]')
parser.add_argument('--nPxPup', default=128, type=int, help='Pupil Resolution')
parser.add_argument('--zModes', default=[2,3], type=int, help='Reconstruction Zernikes')
wfs = parser.parse_args()

wfs.fovInPixel    = wfs.nPxPup*2*wfs.samp 
wfs.pupil = CreateTelescopePupil(wfs.nPxPup,"disc")
wfs.jModes = torch.arange(wfs.zModes[0], wfs.zModes[1]+1)
wfs.pupilLogical = wfs.pupil!=0
wfs.modes = CreateZernikePolynomials(wfs)
wfs.amplitude = 0.1 #small for low noise systems
#wfs.ModPhasor = CreateModulationPhasor(wfs)


#%% ############# Fourier Phase

from phaseGenerators import *

nLenslet      = 16                 # plens res
resAO         = 2*nLenslet+1       # AO resolution           
L0            = 25
fR0           = 1
noiseVariance = 0.7
n_lvl         = 0.1
D             = wfs.D
nTimes        = wfs.fovInPixel/resAO



sub_fold = "S{}_R{}_Z{}-{}_D{:d}".format(wfs.samp,wfs.nPxPup,wfs.zModes[0],wfs.zModes[1],D)
train_fold = main_fold + sub_fold + "/train"
val_fold   = main_fold + sub_fold + "/val"

if not os.path.exists(train_fold):
        os.makedirs(train_fold)
if not os.path.exists(val_fold):
        os.makedirs(val_fold)


a = urand(Zs[0], Zs[1])
#%% Start loop
t0 = time.time()
for k in range(tData):
    Zgt            = a.sample((len(wfs.jModes),1))
    Zgt = Zgt.type(torch.float32)
    phaseMap = torch.reshape(torch.matmul(torch.tensor(wfs.modes).type(torch.float32),Zgt),(wfs.nPxPup,wfs.nPxPup))
    phaseMap = phaseMap.cpu()
    Zgt = Zgt.cpu()
    name = train_fold + "/data_{}.mat".format(k)
    scio.savemat(name, {'x': phaseMap.numpy(),'Zgt': Zgt.numpy()})
    if k%100 == 0:
        t1 = time.time()
        print("Train data ({}/{}) created Avg. time per data {:.2f}ms".format(k,tData,(t1-t0)*10))
        t0 = time.time()
        
t0 = time.time()
for k in range(vData):
    Zgt            = a.sample((len(wfs.jModes),1))
    Zgt = Zgt.type(torch.float64)
    phaseMap = torch.reshape(torch.matmul(torch.tensor(wfs.modes),Zgt),(wfs.nPxPup,wfs.nPxPup))
    phaseMap = phaseMap.cpu()
    Zgt = Zgt.cpu()
    name = val_fold + "/data_{}.mat".format(k)
    scio.savemat(name, {'x': phaseMap.numpy(),'Zgt': Zgt.numpy()})
    if k%100 == 0:
        t1 = time.time()
        print("Validation data ({}/{}) created Avg. time per data {:.2f}ms".format(k,vData,(t1-t0)*10)) 
        t0 = time.time()