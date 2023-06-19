import sys
import os
sys.path.insert(0, '..')

from torch.utils.data import DataLoader
import importlib
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
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
import h5py
from types import SimpleNamespace
import matplotlib.pyplot as plt
matplotlib.interactive(True)

date = datetime.date.today()  
os.chdir("../")




parser = argparse.ArgumentParser(description='Settings, Training and Pyramid Wavefron Sensor parameters')


parser.add_argument('--mods', default=[0,1,2], type=eval, help='Pyramid modulation')
parser.add_argument('--samp', default=2, type=int, help='Over sampling for fourier')
parser.add_argument('--D', default=8, type=int, help='Telescope Diameter [m]')
parser.add_argument('--nPxPup', default=128, type=int, help='Pupil Resolution')
parser.add_argument('--rooftop', default=[0,0], type=eval,help='Pyramid rooftop (as in OOMAO)')
parser.add_argument('--alpha', default=np.pi/2, type=float,help='Pyramid angle (as in OOMAO)')
parser.add_argument('--zModes', default=[2,36], type=eval, help='Reconstruction Zernikes')
parser.add_argument('--ZernikeUnits', default=1, type=float,help='Zernike units (1 for normalized)')
parser.add_argument('--ReadoutNoise', default=0, type=float)
parser.add_argument('--PhotonNoise', default=0, type=float)
parser.add_argument('--nPhotonBackground', default=0, type=float)
parser.add_argument('--quantumEfficiency', default=1, type=float)

parser.add_argument('--D_r0', default=[90,1], type=eval, help='Range of r0 to create')
parser.add_argument('--datapoints', default=10, type=int, help='r0 intervals')
parser.add_argument('--data_batch', default=50, type=int, help='r0 intervals')
parser.add_argument('--dperR0', default=10000, type=int, help='test per datapoint')

parser.add_argument('--models', nargs='+',default=['modelFast'])
parser.add_argument('--checkpoints', nargs='+',default=
                    ['./model/nocap/DE_only/checkpoint/PyrNet_epoch_76.pth'])
parser.add_argument('--saveMats', default="../Matlab/ComputeResults/paper/Fig4A/", type=str)

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


model =[]
for k in range(len(wfs.models)):
    method = importlib.import_module("model_scripts."+wfs.models[k])
    single_model = method.PyrModel(wfs).cuda() 
    checkpoint = torch.load(wfs.checkpoints[k])
    single_model.load_state_dict(checkpoint.state_dict())
    single_model.eval()
    model.append(single_model)



os.makedirs(wfs.saveMats, exist_ok=True)

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

def compute_rmse(vector1, vector2):
    mse = torch.mean((vector1 - vector2) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.detach().cpu().numpy()


Results = []
start = time.time()
for mod in tqdm(wfs.mods,
                                    desc ="Modulation        ",colour="yellow",
                                    total=len(wfs.mods),
                                    ascii=' 123456789═'):
    wfs.modulation    = mod
    wfs.ModPhasor = CreateModulationPhasorCuda(wfs)

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


    ZFull = []
    for k in range(len(wfs.models)+1):
        vsize = np.zeros((wfs.dperR0//wfs.data_batch,wfs.datapoints))
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
    stdZ.append(np.std(ZFull[0],axis=0))     
    stdZ.append(np.std(ZFull[1],axis=0))  


    Dr0ax = Dr0Vector.cpu().numpy()

    RMSEpyr = np.zeros((2,wfs.datapoints))
    RMSEpyr[0,:] = np.mean(ZFull[0],axis=0)
    RMSEpyr[1,:] = np.std(ZFull[0],axis=0)

    RMSEdpwfs = np.zeros((2,wfs.datapoints))
    RMSEdpwfs[0,:] = np.mean(ZFull[1],axis=0)
    RMSEdpwfs[1,:] = np.std(ZFull[1],axis=0)

    INFO = {}
    INFO['D_R0s'] = Dr0ax
    INFO['modulation'] = mod

    struct = {}
    struct['RMSEpyr'] = RMSEpyr
    struct['RMSEdpwfs'] = RMSEdpwfs
    struct['INFO'] = INFO
    Results.append(struct)

end = time.time()

cal_time = end-start
print(f"r0 Figure 4.A completed time({cal_time}) seg for {wfs.datapoints*wfs.dperR0*len(wfs.mods)} total data process.")



sio.savemat(wfs.saveMats+"r0PerformanceFig4A.mat", {'Results': Results},oned_as='row')