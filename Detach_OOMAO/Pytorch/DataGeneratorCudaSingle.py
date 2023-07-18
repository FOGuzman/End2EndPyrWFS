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
import ast
from torch.autograd import Variable
from tqdm import tqdm
from functions.oomao_functions import *
from functions.phaseGeneratorsCuda import *
from math import sqrt, pi
import random

main_fold = "./dataset/dataset_exp/"



parser = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')

parser.add_argument('--modulation', default=0, type=int, help='Pyramid modulation')
parser.add_argument('--samp', default=2, type=int, help='Sampling')
parser.add_argument('--D', default=3, type=int, help='Telescope Diameter [m]')
parser.add_argument('--nPxPup', default=560, type=int, help='Pupil Resolution')
parser.add_argument('--rooftop', default=[0,0], type=eval)
parser.add_argument('--Dr0', default=35, type=int, help='Range of D/r0 to create')
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--alpha', default=pi/2, type=float)
parser.add_argument('--zModes', default=[2,66], type=eval, help='Reconstruction Zernikes')
parser.add_argument('--training_data', default=100, type=int, help='Amount of training data')
# Precalculations
wfs = parser.parse_args()
wfs.fovInPixel    = wfs.nPxPup*2*wfs.samp 
wfs.pupil = CreateTelescopePupil(wfs.nPxPup,"disc")
wfs.pyrMask = createPyrMask(wfs)
wfs.jModes = torch.arange(wfs.zModes[0], wfs.zModes[1]+1)
wfs.pupilLogical = wfs.pupil!=0
wfs.modes = CreateZernikePolynomials(wfs)
wfs.amplitude = 0.2 #small for low noise systems
wfs.modulation = 0    
wfs.ModPhasor = CreateModulationPhasor(wfs)


# To Cuda
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

r0 = wfs.D/wfs.Dr0
tData = wfs.training_data

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=wfs.gpu
n_gpu = torch.cuda.device_count()
print(torch.cuda.is_available())
print('The number of GPU is {} using {}'.format(n_gpu,wfs.gpu))
#%% ############# Fourier Phase

nLenslet      = torch.tensor(16).cuda()                 # plens res
resAO         = 2*nLenslet+1       # AO resolution          
L0            = torch.tensor(25).cuda()   
fR0           = torch.tensor(1).cuda()   
noiseVariance = torch.tensor(0.7).cuda()   
n_lvl         = torch.tensor(0.2).cuda()   
nTimes        = wfs.fovInPixel/resAO



name = "Dr0{}_S{}_R{}_Z{}-{}_D{:d}.mat".format(wfs.Dr0,wfs.samp,wfs.nPxPup,wfs.zModes[0],wfs.zModes[1],wfs.D)

if not os.path.exists(main_fold):
        os.makedirs(main_fold)


# Open a file with access mode 'a'
file_object = open(main_fold  +"/Dataset parameters.txt", 'a')
# Append 'hello' at the end of file
file_object.write("-- Physical parameters --\n")
file_object.write("Modulation             = {}\n".format(wfs.modulation))
file_object.write("Sampling factor        = {}\n".format(wfs.samp))
file_object.write("Sensor resolution      = {}\n".format(wfs.nPxPup))
file_object.write("Zernikes used          = {}\n".format(wfs.zModes))
file_object.write("Diameter               = {}\n".format(wfs.D))
file_object.write("D/r0                   = {}\n".format(wfs.Dr0))
file_object.write("r0                     = {}\n".format(r0))
# Close the file
file_object.close()   
print("Log done!")      


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
atm = GetTurbulenceParameters(wfs,resAO,nLenslet,r0,L0,fR0,noiseVariance,nTimes,n_lvl)
phaseMap,Zgt = GetPhaseMapAndZernike(atm,CM,wfs.training_data)
phaseMap = torch.squeeze(phaseMap).permute(1,2,0)
scio.savemat(main_fold+name, {'x': phaseMap.cpu().numpy(),'Zgt': Zgt.cpu().numpy()})
        
