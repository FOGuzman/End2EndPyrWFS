import sys
import os
sys.path.insert(0, '..')

import importlib
import torch
import time
import datetime
import numpy as np
import argparse
import scipy.io as sio
from functions.oomao_functions import *
from functions.phaseGeneratorsCuda import *
from functions.utils import *
from functions.Propagators import *
from tqdm.auto import tqdm
matplotlib.interactive(True)

date = datetime.date.today()  
os.chdir("../")


parser = argparse.ArgumentParser(description='Settings, Training and Pyramid Wavefron Sensor parameters')

parser.add_argument('--mods', default=[0], type=eval, help='Pyramid modulation')
parser.add_argument('--samp', default=2, type=int, help='Over sampling for fourier')
parser.add_argument('--D', default=8, type=int, help='Telescope Diameter [m]')
parser.add_argument('--nPxPup', default=128, type=int, help='Pupil Resolution')
parser.add_argument('--rooftop', default=[0,0], type=eval,help='Pyramid rooftop (as in OOMAO)')
parser.add_argument('--alpha', default=np.pi/2, type=float,help='Pyramid angle (as in OOMAO)')
parser.add_argument('--zModes', default=[2,66], type=eval, help='Reconstruction Zernikes')
parser.add_argument('--ZernikeUnits', default=1, type=float,help='Zernike units (1 for normalized)')
parser.add_argument('--ReadoutNoise', default=0, type=float)
parser.add_argument('--PhotonNoise', default=0, type=float)
parser.add_argument('--nPhotonBackground', default=0, type=float)
parser.add_argument('--quantumEfficiency', default=1, type=float)
parser.add_argument('--PupilMask', default=None, type=str)

parser.add_argument('--D_r0', default=[50,1], type=eval, help='Range of r0 to create')
parser.add_argument('--datapoints', default=11, type=int, help='r0 intervals')
parser.add_argument('--data_batch', default=200, type=int, help='r0 intervals')
parser.add_argument('--dperR0', default=10000, type=int, help='test per datapoint')

parser.add_argument('--models', nargs='+',default=['modelFast','modelFastPupilPlane'])
parser.add_argument('--checkpoints', nargs='+',default=
                    ['../Preconditioners/original.mat',
                     '../Pytorch/training_results/Paper/06-07-2023/pupil_15-40/checkpoint/PyrNet_epoch_99.pth'])
parser.add_argument('--saveMats', default="../Matlab/ComputeResults/paper/Fig11/", type=str)



# Precalculations
wfs = parser.parse_args()
wfs.fovInPixel    = wfs.nPxPup*2*wfs.samp 
wfs.pupil = CreateTelescopePupil(wfs.nPxPup,"disc")
wfs.pyrMask = createPyrMask(wfs)
wfs.jModes = torch.arange(wfs.zModes[0], wfs.zModes[1]+1)
wfs.pupilLogical = wfs.pupil!=0
wfs.modes = CreateZernikePolynomials(wfs)
wfs.amplitude = 0.1 #small for low noise systems
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

    if wfs.checkpoints[k][-3:] == 'pth':
        checkpoint = torch.load(wfs.checkpoints[k])
        single_model.load_state_dict(checkpoint.state_dict())

    if wfs.checkpoints[k][-3:] == 'mat':    
        checkpoint = sio.loadmat(wfs.checkpoints[k])
        OL1 = torch.nn.Parameter(torch.tensor(checkpoint['OL1']).float().cuda())
        single_model.prop.OL1 = OL1

    single_model.eval()
    model.append(single_model)



os.makedirs(wfs.saveMats, exist_ok=True)

nLenslet      = torch.tensor(16).cuda()                 # plens res
resAO         = 2*nLenslet+1       # AO resolution          
L0            = torch.tensor(25).cuda()   
fR0           = torch.tensor(1).cuda()   
noiseVariance = torch.tensor(0.7).cuda()   
n_lvl         = torch.tensor(0.2).cuda()   
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
    sp = I4Q/spnorm-I_0

    I4Q = Prop2VanillaPyrWFS_torch(-z,wfs)
    smnorm = UNZ(UNZ(UNZ(torch.sum(torch.sum(torch.sum(I4Q,dim=-1),dim=-1),dim=-1),-1),-1),-1)
    sm = I4Q/smnorm-I_0

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

    RMSEdpwfs2 = np.zeros((2,wfs.datapoints))
    RMSEdpwfs2[0,:] = np.mean(ZFull[2],axis=0)
    RMSEdpwfs2[1,:] = np.std(ZFull[2],axis=0)


    INFO = {}
    INFO['D_R0s'] = Dr0ax
    INFO['modulation'] = mod

    struct = {}
    struct['RMSEpyr'] = RMSEpyr
    struct['RMSEdpwfs'] = RMSEdpwfs 
    struct['RMSEdpwfs2'] = RMSEdpwfs2  
    struct['INFO'] = INFO
    Results.append(struct)

end = time.time()

cal_time = end-start
print(f"r0 Figure 11 completed time({cal_time}) seg for {wfs.datapoints*wfs.dperR0*len(wfs.mods)} total data process.")



sio.savemat(wfs.saveMats+"r0PerformanceFig11A.mat", {'Results': Results},oned_as='row')


########## NOISY TEST
wfs.ReadoutNoise = 1
ResultsN = []

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
    sp = I4Q/spnorm-I_0

    I4Q = Prop2VanillaPyrWFS_torch(-z,wfs)
    smnorm = UNZ(UNZ(UNZ(torch.sum(torch.sum(torch.sum(I4Q,dim=-1),dim=-1),dim=-1),-1),-1),-1)
    sm = I4Q/smnorm-I_0

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

            if wfs.PhotonNoise == 1:
                Ip = AddPhotonNoise(Ip,wfs)          
            #Read out noise 
            if wfs.ReadoutNoise != 0:
                Ip = Ip + torch.normal(0,wfs.ReadoutNoise,size=Ip.shape).cuda() 

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

    RMSEdpwfs2 = np.zeros((2,wfs.datapoints))
    RMSEdpwfs2[0,:] = np.mean(ZFull[2],axis=0)
    RMSEdpwfs2[1,:] = np.std(ZFull[2],axis=0)


    INFO = {}
    INFO['D_R0s'] = Dr0ax
    INFO['modulation'] = mod

    struct = {}
    struct['RMSEpyr'] = RMSEpyr
    struct['RMSEdpwfs'] = RMSEdpwfs 
    struct['RMSEdpwfs2'] = RMSEdpwfs2  
    struct['INFO'] = INFO
    Results.append(struct)

end = time.time()

cal_time = end-start
print(f"r0 Figure 11 completed time({cal_time}) seg for {wfs.datapoints*wfs.dperR0*len(wfs.mods)} total data process.")



sio.savemat(wfs.saveMats+"r0PerformanceFig11B.mat", {'ResultsN': ResultsN},oned_as='row')

