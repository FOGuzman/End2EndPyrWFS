from torch.utils.data import DataLoader
import importlib
import torch.optim as optim
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
import argparse
from torch.autograd import Variable
from tqdm import tqdm
from functions.loadData import Imgdataset
from functions.oomao_functions import *
from functions.phaseGenerators import *
from functions.customLoss      import RMSE
from functions.utils import *
from functions.setupFolders import setupFolders
from torch.utils.tensorboard import SummaryWriter


date = datetime.date.today()  

parser = argparse.ArgumentParser(description='Settings, Training and Pyramid Wavefron Sensor parameters')
parser.add_argument('--modulation', default=0, type=int, help='Pyramid modulation')
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

parser.add_argument('--model', default="modelFast", type=str)
parser.add_argument('--batchSize', default=1, type=int, help='Batch size for training')
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--Epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--checkpoint', default=None, type=str)
parser.add_argument('--verbose', action='store_true',help='plot each validation')
parser.add_argument('--experimentName', default="exp_exmaple", type=str)

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
print('The number of GPU is {} using {}'.format(n_gpu,wfs.gpu))

## Setup folders
paths = setupFolders(wfs)


nEpochs    = wfs.Epochs
lr         = wfs.learning_rate
zu         = wfs.ZernikeUnits


# Model definition
method = importlib.import_module("model_scripts."+wfs.model)
PyrNet = method.PyrModel(wfs).cuda()              

# Load Checkpoint
if wfs.checkpoint is not None:
        PyrNet = torch.load(wfs.checkpoint)
        PyrNet = PyrNet.module if hasattr(PyrNet, "module") else PyrNet
        print("Checkpoint loaded successfully!")
else:
        print("Training from scrach.")

dataset = Imgdataset(paths.train_fold)
train_data_loader = DataLoader(dataset=dataset, batch_size=wfs.batchSize, shuffle=True)


loss = RMSE()
loss.cuda()            


# Enable dataparallelism (if mor that 1 GPU available)
if n_gpu > 1:
    PyrNet = torch.nn.DataParallel(PyrNet)
    
## Train

# testing loop
def test(epoch, model,paths):
    test_list = os.listdir(paths.val_fold)
    rmse_cnn = torch.zeros(len(test_list))
    Ypyr_res = None
    Ygt_res = None
    for i in range(len(test_list)):
        rmse_1 = 0  
        datamat = scio.loadmat(paths.val_fold + '/' + test_list[i])       
        Ygt = datamat['Zgt']
        Ygt = torch.from_numpy(Ygt).cuda().float()*zu
        Ygt = torch.transpose(Ygt,0,1)
        phaseMap = datamat['x']
        phaseMap = torch.from_numpy(phaseMap).cuda().float()
        phaseMap = torch.unsqueeze(torch.unsqueeze(phaseMap,0),0)
        with torch.no_grad():
                     
            Ypyr = model(phaseMap)*zu
            rmse_1 = torch.sqrt(torch.mean((Ygt-Ypyr)**2)) 
            rmse_cnn[i] = rmse_1

            a = test_list[i]
            name = paths.result_path + '/PyrNet_' + a[0:len(a) - 4] + '_{}_{:.4f}'.format(epoch, rmse_cnn[i]) + '.mat'
            if Ypyr_res is not None:
                Ypyr_res = torch.concat([Ypyr_res,Ypyr.cpu()],1)
                Ygt_res = torch.concat([Ygt_res,Ygt.cpu()],1)
            else:
                Ypyr_res = Ypyr.cpu()
                Ygt_res = Ypyr.cpu()
            
            
    prtname = "Epochs mean result: RMSE -- {:.4f}".format(torch.mean(rmse_cnn))   
    print(prtname)

    scio.savemat(name, {'Ypyr': Ypyr_res.numpy(),'Ygt': Ygt_res.numpy()})
    OL1_trained = PyrNet.state_dict()['prop.OL1'].cpu()

    #Add things to tensorboard
    im_out = map_tensor_to_range(torch.fft.fftshift(OL1_trained))
    writer.add_scalar("RMSE",torch.mean(rmse_cnn),epoch)
    writer.add_image("DE trained",im_out,epoch,dataformats='HW')
    fig = plot_summary(torch.fft.fftshift(OL1_trained),Ypyr_res.numpy()[:,-1],Ygt_res.numpy()[:,-1],prtname,phaseMap,wfs)
    writer.add_figure("status",fig,epoch,close=True)

    #If want to plot
    if wfs.verbose:
        plt.show(block=False)
        plt.pause(1) 
        

    scio.savemat(paths.de_path + "OL1_R{}_M{}_RMSE{:.4}_Epoch_{}.mat".format(
        np.int(wfs.nPxPup),np.int(wfs.modulation),torch.mean(rmse_cnn),epoch) 
                 , {'OL1': OL1_trained.numpy()})


# training loop
def train(epoch, model, lr,paths):
    epoch_loss = 0
    begin = time.time()

    optimizer_g = optim.AdamW([{'params': model.parameters()}], lr=lr)
    for iteration, batch in tqdm(enumerate(train_data_loader),
                                 desc ="Training... ",colour="red",
                                 total=len(train_data_loader)//wfs.batchSize,
                                 ascii=' 123456789═'):
        Ygt = Variable(batch[0])
        Ygt = Ygt.cuda().float()*zu
        Ygt = torch.transpose(Ygt,0,1)
        phaseMap = Variable(batch[1])
        phaseMap = phaseMap.cuda().float()
        phaseMap = torch.unsqueeze(phaseMap,1)

        optimizer_g.zero_grad()
        Ypyr = model(phaseMap)*zu
        Loss1 = loss(Ypyr,Ygt)
        Loss1.backward()
        optimizer_g.step()
        epoch_loss += Loss1.data

        if (iteration % 100 == 0):
            writer.add_scalar("loss",Loss1.item(),epoch*len(train_data_loader) + iteration)

    model = model.module if hasattr(model, "module") else model
    test(epoch,model.eval(),paths)
    end = time.time()
    print("===> Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch, epoch_loss / len(train_data_loader)),
          "  time: {:.2f}".format(end - begin))
    
def checkpoint(epoch, paths):
    model_out_path =  paths.model_path + "PyrNet_epoch_{}.pth".format(epoch)
    torch.save(PyrNet, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))    
    



## Main
GenerateLog(date,paths,wfs,loss,"update")
writer = SummaryWriter(paths.tb_path)


for epoch in range(nEpochs):
    train(epoch, PyrNet, lr,paths)
    if (epoch % 5 == 0) and (epoch < 100):
        lr = lr * 0.95
        print("Learning rate changeg to {:.6}".format(lr)) 
    if (epoch % 1 == 0 or epoch > 50):
        PyrNet = PyrNet.module if hasattr(PyrNet, "module") else PyrNet
        checkpoint(epoch, paths)
    if n_gpu > 1:
        PyrNet = torch.nn.DataParallel(PyrNet)    
 
        
writer.close()