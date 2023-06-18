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
from functions.phaseGenerators import *
from math import sqrt, pi
import random

main_fold = "./dataset/"



parser = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')

parser.add_argument('--modulation', default=0, type=int, help='Pyramid modulation')
parser.add_argument('--samp', default=2, type=int, help='Sampling')
parser.add_argument('--D', default=3, type=int, help='Telescope Diameter [m]')
parser.add_argument('--nPxPup', default=128, type=int, help='Pupil Resolution')
parser.add_argument('--rooftop', default=[0,0], type=eval)
parser.add_argument('--Dr0', default=[15,40], type=eval, help='Range of D/r0 to create')
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--alpha', default=pi/2, type=float)
parser.add_argument('--zModes', default=[2,24], type=eval, help='Reconstruction Zernikes')
parser.add_argument('--training_data', default=100, type=int, help='Amount of training data')
parser.add_argument('--validation_data', default=10, type=int, help='Amount of validation data')
wfs = parser.parse_args()
wfs.Dr0 = np.array(wfs.Dr0)
wfs.r0 = np.round(wfs.D/wfs.Dr0[::-1], 2) # flipped Dr0 to obtain the r0 in a increasing pattern
wfs.fovInPixel    = wfs.nPxPup*2*wfs.samp 
wfs.pupil = CreateTelescopePupil(wfs.nPxPup,"disc")
wfs.pyrMask = createPyrMask(wfs)
wfs.jModes = torch.arange(wfs.zModes[0], wfs.zModes[1]+1)
wfs.pupilLogical = wfs.pupil!=0
wfs.modes = CreateZernikePolynomials(wfs)
wfs.amplitude = 0.1 #small for low noise systems
wfs.ModPhasor = CreateModulationPhasor(wfs)


tData = wfs.training_data
vData = wfs.validation_data
Rs = wfs.r0

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=wfs.gpu
n_gpu = torch.cuda.device_count()
print(torch.cuda.is_available())
print('The number of GPU is {} using {}'.format(n_gpu,wfs.gpu))
#%% ############# Fourier Phase

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


# Open a file with access mode 'a'
file_object = open(main_fold + sub_fold +"/Dataset parameters.txt", 'a')
# Append 'hello' at the end of file
file_object.write("-- Physical parameters --\n")
file_object.write("Modulation             = {}\n".format(wfs.modulation))
file_object.write("Sampling factor        = {}\n".format(wfs.samp))
file_object.write("Sensor resolution      = {}\n".format(wfs.nPxPup))
file_object.write("Zernikes used          = {}\n".format(wfs.zModes))
file_object.write("Diameter               = {}\n".format(wfs.D))
file_object.write("D/r0                   = {}\n".format(wfs.Dr0))
file_object.write("r0                     = {}\n".format(wfs.r0))
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
    scio.savemat(name, {'x': phaseMap.numpy(),'Zgt': Zgt.numpy(),'r0':r0})
    if k%100 == 0:
        t1 = time.time()
        print("Validation data ({}/{}) created Avg. time per data {:.2f}ms".format(k,vData,(t1-t0)*10)) 
        t0 = time.time()
