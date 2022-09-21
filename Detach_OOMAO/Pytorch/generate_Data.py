from loadData import Imgdataset
from torch.utils.data import DataLoader
from models_generalized import kNet
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

tData = 20000
vData = 4000

Rs = [0.3,1]

train_fold = "./dataset/train"
val_fold   = "./dataset/val"

if not os.path.exists(train_fold):
        os.makedirs(train_fold)
if not os.path.exists(val_fold):
        os.makedirs(val_fold)


parser = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')

parser.add_argument('--modulation', default=0, type=int, help='Pyramid modulation')
parser.add_argument('--samp', default=2, type=int, help='Sampling')
parser.add_argument('--nPxPup', default=128, type=int, help='Pupil Resolution')
parser.add_argument('--rooftop', default=[0,0], type=float)
parser.add_argument('--alpha', default=pi/2, type=float)
parser.add_argument('--zModes', default=[2,50], type=int, help='Reconstruction Zernikes')
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
L0            = 20
fR0           = 1
noiseVariance = 0.7
n_lvl         = 0.1
D             = 8
nTimes        = wfs.fovInPixel/resAO


for k in range(tData):
    r0            = random.uniform(Rs[0], Rs[1])
    phaseMap,Zgt = GenerateFourierPhaseXY(r0,L0,D,resAO,nLenslet,nTimes,n_lvl,noiseVariance,wfs)
    phaseMap = torch.unsqueeze(torch.tensor(phaseMap),0)
    name = train_fold + "/data_{}.mat".format(k)
    scio.savemat(name, {'x': phaseMap.numpy(),'Zgt': Zgt.numpy()})
    if k%100 == 0:
        print("Train data ({}/{}) created".format(k+1,tData))
    

for k in range(vData):
    r0            = random.uniform(Rs[0], Rs[1])
    phaseMap,Zgt = GenerateFourierPhaseXY(r0,L0,D,resAO,nLenslet,nTimes,n_lvl,noiseVariance,wfs)
    phaseMap = torch.unsqueeze(torch.tensor(phaseMap),0)
    name = val_fold + "/data_{}.mat".format(k)
    scio.savemat(name, {'x': phaseMap.numpy(),'Zgt': Zgt.numpy()})
    if k%100 == 0:
        print("Validation data ({}/{}) created".format(k+1,vData))    
