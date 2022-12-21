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
from oomao_functions import *
from phaseGenerators import *
from math import sqrt, pi
import random



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
n_gpu = torch.cuda.device_count()
print(torch.cuda.is_available())
print('The number of GPU is {}'.format(n_gpu))

tData = 5000
vData = 500
Rs = [0.08,0.2]
main_fold = "./dataset/"



parser = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')

parser.add_argument('--modulation', default=2, type=int, help='Pyramid modulation')
parser.add_argument('--samp', default=2, type=int, help='Sampling')
parser.add_argument('--D', default=3, type=int, help='Telescope Diameter [m]')
parser.add_argument('--nPxPup', default=64, type=int, help='Pupil Resolution')
parser.add_argument('--rooftop', default=[0,0], type=float)
parser.add_argument('--alpha', default=pi/2, type=float)
parser.add_argument('--zModes', default=[2,24], type=int, help='Reconstruction Zernikes')
wfs = parser.parse_args()

wfs.fovInPixel    = wfs.nPxPup*2*wfs.samp 
wfs.pupil = CreateTelescopePupil(wfs.nPxPup,"disc")
wfs.pyrMask = createPyrMask(wfs)
wfs.jModes = torch.arange(wfs.zModes[0], wfs.zModes[1]+1)
wfs.pupilLogical = wfs.pupil!=0
wfs.modes = CreateZernikePolynomials(wfs)
wfs.amplitude = 0.1 #small for low noise systems
wfs.ModPhasor = CreateModulationPhasor(wfs)


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


IM = None
    #%% Control matrix
for k in range(len(wfs.jModes)):
    imMode = torch.reshape(torch.tensor(wfs.modes[:,k]),(wfs.nPxPup,wfs.nPxPup))
    Zv = torch.unsqueeze(torch.reshape(imMode,[-1]),1)
    if IM is not None:
        IM = torch.concat([IM,Zv],1)
    else:
        IM = Zv
    
    
CM = torch.linalg.pinv(IM)

#%% Start loop
t0 = time.time()
for k in range(tData):
    r0            = random.uniform(Rs[0], Rs[1])
    atm = GetTurbulenceParameters(wfs,resAO,nLenslet,r0,L0,fR0,noiseVariance,nTimes,n_lvl)
    psdAO_mean = torch.tensor(atm['psdAO_mean'])
    N = torch.tensor(atm['N'],dtype=torch.float64)
    fourierSampling = torch.tensor(atm['fourierSampling'])
    idx = torch.tensor(atm['idx'])
    pupil = torch.tensor(atm['pupil'])
    nPxPup = torch.tensor(atm['nPxPup'])
    phaseMap,Zgt = GetPhaseMapAndZernike_CUDA(psdAO_mean.cuda(),N.cuda(),fourierSampling.cuda(),idx.cuda(),pupil.cuda(),nPxPup.cuda(),CM.cuda())
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
    r0            = random.uniform(Rs[0], Rs[1])
    atm = GetTurbulenceParameters(wfs,resAO,nLenslet,r0,L0,fR0,noiseVariance,nTimes,n_lvl)
    psdAO_mean = torch.tensor(atm['psdAO_mean'])
    N = torch.tensor(atm['N'],dtype=torch.float64)
    fourierSampling = torch.tensor(atm['fourierSampling'])
    idx = torch.tensor(atm['idx'])
    pupil = torch.tensor(atm['pupil'])
    nPxPup = torch.tensor(atm['nPxPup'])
    phaseMap,Zgt = GetPhaseMapAndZernike_CUDA(psdAO_mean.cuda(),N.cuda(),fourierSampling.cuda(),idx.cuda(),pupil.cuda(),nPxPup.cuda(),CM.cuda())
    phaseMap = phaseMap.cpu()
    Zgt = Zgt.cpu()
    name = val_fold + "/data_{}.mat".format(k)
    scio.savemat(name, {'x': phaseMap.numpy(),'Zgt': Zgt.numpy()})
    if k%100 == 0:
        t1 = time.time()
        print("Validation data ({}/{}) created Avg. time per data {:.2f}ms".format(k,vData,(t1-t0)*10)) 
        t0 = time.time()