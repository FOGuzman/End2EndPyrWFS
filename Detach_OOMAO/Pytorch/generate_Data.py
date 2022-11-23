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

tData = 10000
vData = 1000
Rs = [0.2,1.2]
main_fold = "./dataset/"



parser = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')

parser.add_argument('--modulation', default=1, type=int, help='Pyramid modulation')
parser.add_argument('--samp', default=2, type=int, help='Sampling')
parser.add_argument('--nPxPup', default=128, type=int, help='Pupil Resolution')
parser.add_argument('--rooftop', default=[0,0], type=float)
parser.add_argument('--alpha', default=pi/2, type=float)
parser.add_argument('--zModes', default=[2,26], type=int, help='Reconstruction Zernikes')
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
D             = 8
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

for k in range(tData):
    r0            = random.uniform(Rs[0], Rs[1])
    phaseMap,Zgt = GenerateFourierPhaseXY(r0,L0,D,resAO,nLenslet,nTimes,n_lvl,noiseVariance,CM,wfs)
    phaseMap = torch.unsqueeze(torch.tensor(phaseMap),0)
    name = train_fold + "/data_{}.mat".format(k)
    scio.savemat(name, {'x': phaseMap.numpy(),'Zgt': Zgt.numpy()})
    if k%100 == 0:
        print("Train data ({}/{}) created".format(k+1,tData))
    

for k in range(vData):
    r0            = random.uniform(Rs[0], Rs[1])
    phaseMap,Zgt = GenerateFourierPhaseXY(r0,L0,D,resAO,nLenslet,nTimes,n_lvl,noiseVariance,CM,wfs)
    phaseMap = torch.unsqueeze(torch.tensor(phaseMap),0)
    name = val_fold + "/data_{}.mat".format(k)
    scio.savemat(name, {'x': phaseMap.numpy(),'Zgt': Zgt.numpy()})
    if k%100 == 0:
        print("Validation data ({}/{}) created".format(k+1,vData))    
