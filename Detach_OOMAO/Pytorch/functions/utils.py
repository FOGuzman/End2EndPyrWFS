import os
import datetime
import matplotlib.pyplot as plt
import math
from mpmath import *
import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
matplotlib.interactive(False)

def GenerateLog(date,paths,wfs,loss,mode):
    
    
       
 d = date.strftime("%b-%d-%Y")
 log_name = paths.log_path +"/"+ d +".txt"  
 if not os.path.exists(paths.log_path):
        os.makedirs(paths.log_path) 
 if not os.path.exists(log_name) or mode=="update":
        open(log_name, 'w')        
        
 # Open a file with access mode 'a'
 file_object = open(log_name, 'a')
 # Append 'hello' at the end of file
 file_object.write("-- Physical parameters --\n")
 file_object.write("Modulation             = {}\n".format(wfs.modulation))
 file_object.write("Sampling factor        = {}\n".format(wfs.samp))
 file_object.write("Sensor resolution      = {}\n".format(wfs.nPxPup))
 file_object.write("Pyramid rooftop        = {}\n".format(wfs.rooftop))
 file_object.write("Pyramid alpha          = {}\n".format(wfs.alpha))
 file_object.write("Zernikes used          = {}\n".format(wfs.zModes))
 file_object.write("Zernikes units         = {}\n".format(wfs.ZernikeUnits))
 file_object.write("Readout noise          = {}\n".format(wfs.ReadoutNoise))
 file_object.write("Photon Noise           = {}\n".format(wfs.PhotonNoise))
 file_object.write("Photons Background     = {}\n".format(wfs.nPhotonBackground))
 file_object.write("Quantum Efficiency     = {}\n".format(wfs.quantumEfficiency))
 file_object.write("--   Network parameters  --\n")
 file_object.write("Model Used             = {}\n".format(wfs.model))
 file_object.write("Batch Size             = {}\n".format(wfs.batchSize))
 file_object.write("Epochs                 = {}\n".format(wfs.Epochs))
 file_object.write("Initial Learning Rate  = {}\n".format(wfs.Epochs))
 file_object.write("Result path            = {}\n".format(paths.result_path))
 file_object.write("Loss function          = {}\n".format(loss))
 
 # Close the file
 file_object.close()   
 print("Log done!")      


def map_tensor_to_range(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    if max_val == min_val:
         min_val = 0
    
    scaled_tensor = (tensor - min_val) / (max_val - min_val)  # Map tensor to the range 0 to 1
    mapped_tensor = (scaled_tensor * 255).to(torch.uint8)  # Map tensor to the range 0 to 255
    
    return mapped_tensor


def plot_summary(t,yr,ygt,name,phaseMap,wfs):
    plt.close()
    Z_vect = np.arange(wfs.zModes[0],wfs.zModes[1]+1)

    fig, axes = plt.subplot_mosaic("AB;CC",figsize=(10,8))
    line1 = axes["C"].plot(Z_vect,yr,label=wfs.model)
    line2 = axes["C"].plot(Z_vect,ygt,label="Groundtruth")
    axes["C"].legend()
    axes["C"].set_xlabel('Zernike index')
    t = np.array(t)
    t_ = np.squeeze(t)
    
    Pa = axes["A"].imshow(t_, cmap ='jet', interpolation ='nearest', origin ='lower')
    Pb = axes["B"].imshow(torch.squeeze(phaseMap).detach().cpu().numpy(), cmap ='jet', interpolation ='nearest', origin ='lower')
    plt.colorbar(Pa, ax=axes["A"])
    plt.colorbar(Pb, ax=axes["B"])

    fig.suptitle(name)
    axes["A"].title.set_text('Difrractive Element')
    axes["B"].title.set_text('phaseMap example')
    axes["C"].title.set_text('Zernike comparison example')

    return(fig)
