from loadData import Imgdataset
from torch.utils.data import DataLoader
from modelFast import PyrModel,PhaseConstraint
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
from customLoss      import RMSE
from math import sqrt, pi
from utils import *


date = datetime.date.today()  





parser = argparse.ArgumentParser(description='Settings, Training and Pyramid Wavefron Sensor parameters')

parser.add_argument('--modulation', default=0, type=int, help='Pyramid modulation')
parser.add_argument('--samp', default=2, type=int, help='Over sampling for fourier')
parser.add_argument('--D', default=8, type=int, help='Telescope Diameter [m]')
parser.add_argument('--nPxPup', default=128, type=int, help='Pupil Resolution')
parser.add_argument('--rooftop', default=[0,0], type=float,help='Pyramid rooftop (as in OOMAO)')
parser.add_argument('--alpha', default=pi/2, type=float,help='Pyramid angle (as in OOMAO)')
parser.add_argument('--zModes', default=[2,36], type=int, help='Reconstruction Zernikes')
parser.add_argument('--batchSize', default=1, type=int, help='Batch size for training')
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--ReadoutNoise', default=1, type=float)
parser.add_argument('--PhotonNoise', default=0, type=float)
parser.add_argument('--nPhotonBackground', default=0, type=float)
parser.add_argument('--quantumEfficiency', default=1, type=float)


# Precalculations
wfs = parser.parse_args()
wfs.fovInPixel    = wfs.nPxPup*2*wfs.samp 
wfs.pupil = CreateTelescopePupil(wfs.nPxPup,"disc")
wfs.pyrMask = createPyrMask(wfs)
wfs.jModes = torch.arange(wfs.zModes[0], wfs.zModes[1]+1)
wfs.pupilLogical = wfs.pupil!=0
wfs.modes = CreateZernikePolynomials(wfs)
wfs.amplitude = 0.2 #small for low noise systems
wfs.ModPhasor = CreateModulationPhasor(wfs)


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=wfs.gpu
n_gpu = torch.cuda.device_count()
print(torch.cuda.is_available())
print('The number of GPU is {}'.format(n_gpu))

## Network parameters
main_fold = "./dataset/"
sub_fold = "S{}_R{}_Z{}-{}_D{:d}".format(wfs.samp,wfs.nPxPup,wfs.zModes[0],wfs.zModes[1],wfs.D)
train_fold = main_fold + sub_fold + "/train"
val_fold   = main_fold + sub_fold + "/val"
model_path = "./model/nocap/" + sub_fold + "/checkpoint"
result_path = "./results"
log_path   = "./model/nocap/" + sub_fold + "/"
load_train = 0
nEpochs    = 100
lr         = 0.001


# Model definition
PyrNet = PyrModel(wfs)              
PyrNet = PyrModel(wfs).cuda()


dataset = Imgdataset(train_fold)
train_data_loader = DataLoader(dataset=dataset, batch_size=wfs.batchSize, shuffle=True)


loss = RMSE()
loss.cuda()            


# Enable dataparallelism (if mor that 1 GPU available)
if n_gpu > 1:
    PyrNet = torch.nn.DataParallel(PyrNet)
if load_train != 0:
    PyrNet = torch.load(model_path + "/PyrNet_epoch_{}.pth".format(load_train))
    PyrNet = PyrNet.module if hasattr(PyrNet, "module") else PyrNet
    
    
## Train

# testing loop
def test(test_path, epoch, result_path, model):
    test_list = os.listdir(test_path)
    rmse_cnn = torch.zeros(len(test_list))
    Ypyr_res = None
    Ygt_res = None
    for i in range(len(test_list)):
        datamat = scio.loadmat(test_path + '/' + test_list[i])
        
        Ygt = datamat['Zgt']
        Ygt = torch.from_numpy(Ygt).cuda().float()
        Ygt = torch.transpose(Ygt,0,1)
        phaseMap = datamat['x']
        phaseMap = torch.from_numpy(phaseMap).cuda().float()

        with torch.no_grad():

            rmse_1 = 0
            phaseMap = torch.unsqueeze(phaseMap,0)
            Ypyr = model(phaseMap)
            rmse_1 = torch.sqrt(torch.mean((Ygt-Ypyr)**2)) 
            rmse_cnn[i] = rmse_1

            a = test_list[i]
            name = result_path + '/PyrNet_' + a[0:len(a) - 4] + '_{}_{:.4f}'.format(epoch, rmse_cnn[i]) + '.mat'
            if Ypyr_res is not None:
                Ypyr_res = torch.concat([Ypyr_res,Ypyr.cpu()],1)
                Ygt_res = torch.concat([Ygt_res,Ygt.cpu()],1)
            else:
                Ypyr_res = Ypyr.cpu()
                Ygt_res = Ypyr.cpu()
            
            
    prtname = "Diffractive Element result: RMSE -- {:.4f}".format(torch.mean(rmse_cnn))        
    scio.savemat(name, {'Ypyr': Ypyr_res.numpy(),'Ygt': Ygt_res.numpy()})
    print(prtname)
    OL1_trained = PyrNet.state_dict()['prop.OL1'].cpu()
    plot_tensorwt(torch.fft.fftshift(OL1_trained),prtname)
    scio.savemat(model_path + "/OL1_R{}_M{}_RMSE{:.4}_Epoch_{}.mat".format(
        np.int(wfs.nPxPup),np.int(wfs.modulation),torch.mean(rmse_cnn),epoch) 
                 , {'OL1': OL1_trained.numpy()})









# training loop
def train(epoch, result_path, model, lr):
    epoch_loss = 0
    begin = time.time()

    optimizer_g = optim.AdamW([{'params': model.parameters()}], lr=lr)

    for iteration, batch in tqdm(enumerate(train_data_loader)):
        Ygt = Variable(batch[0])
        Ygt = Ygt.cuda().float()
        Ygt = torch.transpose(Ygt,0,1)
        phaseMap = Variable(batch[1])
        phaseMap = phaseMap.cuda().float()
        phaseMap = torch.unsqueeze(phaseMap,1)

        optimizer_g.zero_grad()
        Ypyr = model(phaseMap)
        Loss1 = loss(Ypyr,Ygt)

        Loss1.backward()
        optimizer_g.step()
        epoch_loss += Loss1.data

    model = model.module if hasattr(model, "module") else model
    test(val_fold, epoch, result_path, model.eval())
    end = time.time()
    print("===> Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch, epoch_loss / len(train_data_loader)),
          "  time: {:.2f}".format(end - begin))
    
def checkpoint(epoch, model_path):
    model_out_path =  model_path + '/' + "PyrNet_epoch_{}.pth".format(epoch)
    torch.save(PyrNet, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))    
    



## Main
if not os.path.exists(result_path):
        os.makedirs(result_path)
if not os.path.exists(model_path):
        os.makedirs(model_path)

GenerateLog(date,log_path,result_path,wfs,loss,"update")        
for epoch in range(load_train + 1, load_train + nEpochs + 1):
    train(epoch, result_path, PyrNet, lr)
    if (epoch % 5 == 0) and (epoch < 100):
        lr = lr * 0.95
        print("Learning rate changeg to {:.6}".format(lr)) 
    if (epoch % 1 == 0 or epoch > 50):
        PyrNet = PyrNet.module if hasattr(PyrNet, "module") else PyrNet
        checkpoint(epoch, model_path)
    if n_gpu > 1:
        PyrNet = torch.nn.DataParallel(PyrNet)    
 
        
