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
import scipy.io as sio
from functions.oomao_functions import *
from functions.phaseGeneratorsCuda import *
from functions.customLoss      import RMSE
from functions.utils import *
from functions.Propagators import *
from tqdm.auto import tqdm
import random
from types import SimpleNamespace
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
matplotlib.interactive(True)

date = datetime.date.today()  





parser = argparse.ArgumentParser(description='Settings, Training and Pyramid Wavefron Sensor parameters')


parser.add_argument('--modulation', default=0, type=int, help='Pyramid modulation')
parser.add_argument('--samp', default=2, type=int, help='Over sampling for fourier')
parser.add_argument('--D', default=8, type=int, help='Telescope Diameter [m]')
parser.add_argument('--nPxPup', default=64, type=int, help='Pupil Resolution')
parser.add_argument('--rooftop', default=[0,0], type=eval,help='Pyramid rooftop (as in OOMAO)')
parser.add_argument('--alpha', default=np.pi/2, type=float,help='Pyramid angle (as in OOMAO)')
parser.add_argument('--zModes', default=[2,64], type=eval, help='Reconstruction Zernikes')
parser.add_argument('--ZernikeUnits', default=1, type=float,help='Zernike units (1 for normalized)')
parser.add_argument('--ReadoutNoise', default=0, type=float)
parser.add_argument('--PhotonNoise', default=0, type=float)
parser.add_argument('--nPhotonBackground', default=0, type=float)
parser.add_argument('--quantumEfficiency', default=1, type=float)

parser.add_argument('--D_r0', default=[90,10], type=eval, help='Range of r0 to create')
parser.add_argument('--datapoints', default=10, type=int, help='r0 intervals')
parser.add_argument('--data_batch', default=10, type=int, help='r0 intervals')
parser.add_argument('--dperR0', default=10000, type=int, help='test per datapoint')

parser.add_argument('--models', nargs='+',default=['ConvNeXt_only','modelFast','modelFastplusConvNeXt','modelFastplusConvNeXt'])
parser.add_argument('--checkpoints', nargs='+',default=
                    ['./training_results/EV/ConvNeXt/checkpoint/PyrNet_epoch_99.pth',
                     './training_results/EV/DPWFS/checkpoint/PyrNet_epoch_99.pth',
                     './training_results/EV/ConvNeXt+DE/checkpoint/PyrNet_epoch_99.pth',
                     './training_results/EV/ConvNeXt+DE+hybrid/checkpoint/PyrNet_epoch_99.pth'])



# Precalculations
wfs = parser.parse_args()
wfs.fovInPixel    = wfs.nPxPup*2*wfs.samp 
wfs.pupil = CreateTelescopePupil(wfs.nPxPup,"disc")
wfs.pyrMask = createPyrMask(wfs)
wfs.jModes = torch.arange(wfs.zModes[0], wfs.zModes[1]+1)
wfs.pupilLogical = wfs.pupil!=0
wfs.modes = CreateZernikePolynomials(wfs)
wfs.amplitude = 0.1 #small for low noise systems  
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


model =[]
for k in range(len(wfs.models)):
    method = importlib.import_module("model_scripts."+wfs.models[k])
    single_model = method.PyrModel(wfs).cuda() 
    checkpoint = torch.load(wfs.checkpoints[k])
    single_model.load_state_dict(checkpoint.state_dict())
    single_model.eval()
    model.append(single_model)




nLenslet      = torch.tensor(16).cuda()                 # plens res
resAO         = 2*nLenslet+1       # AO resolution
r0            = torch.tensor(0.11).cuda()            
L0            = torch.tensor(25).cuda()   
fR0           = torch.tensor(1.2).cuda()   
noiseVariance = torch.tensor(0.7).cuda()   
n_lvl         = torch.tensor(0.1).cuda()   
nTimes        = wfs.fovInPixel/resAO


Dr0Vector = torch.linspace(wfs.D_r0[0],wfs.D_r0[1],wfs.datapoints).cuda()
r0Vector = wfs.D/Dr0Vector



CMPhase = torch.linalg.pinv(torch.tensor(wfs.modes))



###### Flat prop      
        # Flat prop
Flat = torch.ones((wfs.nPxPup,wfs.nPxPup))*wfs.pupilLogical
Flat = UNZ(UNZ(Flat,0),0).cuda()        
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


def compute_rmse(vector1, vector2):
    mse = torch.mean((vector1 - vector2) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.detach().cpu().numpy()


ZFull = []
for k in range(len(wfs.models)+1):
    vsize = np.zeros((wfs.dperR0,wfs.datapoints))
    ZFull.append(vsize) 



for r0idx in tqdm(range(0,wfs.datapoints),
                                 desc ="r0 tested         ",colour="red",
                                 total=wfs.datapoints,
                                 ascii=' 123456789═'):
    r0el = r0Vector[r0idx]
    for nridx in tqdm(range(0,wfs.dperR0//wfs.data_batch),
                                 desc ="Datapoints tested ",colour="green",
                                 total=wfs.dperR0//wfs.data_batch,
                                 ascii=' 123456789═'):

        atm = GetTurbulenceParameters(wfs,resAO,nLenslet,r0el,L0,fR0,noiseVariance,nTimes,n_lvl)
        phaseMap,Zgt = GetPhaseMapAndZernike(atm,CMPhase,wfs.data_batch)
        Ip = Prop2VanillaPyrWFS_torch(phaseMap,wfs)
        Inorm = torch.sum(torch.sum(torch.sum(Ip,-1),-1),-1)
        Ip = Ip/UNZ(UNZ(UNZ(Inorm,-1),-1),-1)-I_0

        Zpyr = torch.matmul(CM,torch.transpose(torch.reshape(Ip,[Ip.shape[0],-1]),0,1))

        ZFull[0][nridx,r0idx] = compute_rmse(Zgt,Zpyr)
        Zest = []
        for m in range(len(wfs.models)):
            Z_single = model[m](phaseMap).detach()
            ZFull[m+1][nridx,r0idx] = compute_rmse(Zgt,Z_single)



meanZ = []
stdZ  = []
meanZ.append(np.mean(ZFull[0],axis=0))     
meanZ.append(np.mean(ZFull[1],axis=0)) 
meanZ.append(np.mean(ZFull[2],axis=0)) 
meanZ.append(np.mean(ZFull[3],axis=0)) 
meanZ.append(np.mean(ZFull[4],axis=0))
stdZ.append(np.std(ZFull[0],axis=0))     
stdZ.append(np.std(ZFull[1],axis=0)) 
stdZ.append(np.std(ZFull[2],axis=0)) 
stdZ.append(np.std(ZFull[3],axis=0)) 
stdZ.append(np.std(ZFull[4],axis=0))

Results = []
Dr0ax = Dr0Vector.cpu().numpy()

INFO = {}
INFO['D_R0s'] = Dr0ax
INFO['modulation'] = wfs.modulation
methods= ["PyrWFS","DPWFS","ConvNeXt","ConvNext+DE","ConvNext+DE*"]

for index, method in enumerate(methods):
    RMSE_vec = np.zeros((2,wfs.datapoints))
    RMSE_vec[0,:] = np.mean(ZFull[index],axis=0)
    RMSE_vec[1,:] = np.std(ZFull[index],axis=0)
    struct = {}
    struct['RMSE_vec'] = RMSE_vec
    struct['INFO'] = INFO
    struct['method'] = method
    Results.append(struct)

sio.savemat("./training_results/EV/r0Performance_methods.mat", {'Results': Results},oned_as='row')


fig = plt.figure()



plt.errorbar(Dr0ax, meanZ[0], yerr=stdZ[0], label='PyrWFS')
plt.errorbar(Dr0ax, meanZ[1], yerr=stdZ[1], label='CNN')
plt.errorbar(Dr0ax, meanZ[2], yerr=stdZ[2], label='DPWFS')
plt.errorbar(Dr0ax, meanZ[3], yerr=stdZ[3], label='CNN+DPWFS')
plt.errorbar(Dr0ax, meanZ[3], yerr=stdZ[3], label='CNN+DPWFS*')
plt.gca().invert_xaxis()        
plt.legend()
plt.show(block=True)        