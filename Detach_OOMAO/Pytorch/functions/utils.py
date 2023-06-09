import os
import datetime
import matplotlib.pyplot as plt
import math
from mpmath import *
import matplotlib
import numpy as np

matplotlib.interactive(False)

def GenerateLog(date,log_path,result_path,wfs,loss,mode):
    
    
       
 d = date.strftime("%b-%d-%Y")
 log_name = log_path +"/"+ d +".txt"  
 if not os.path.exists(log_path):
        os.makedirs(log_path) 
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
 file_object.write("Result path            = {}\n".format(result_path))
 file_object.write("Loss function          = {}\n".format(loss))
 
 # Close the file
 file_object.close()   
 print("Log done!")      





def plot_tensorwt(t,yr,ygt,name):
    plt.close()
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(9, 3))
    line1, = ax2.plot(yr[:,-1],label="Estimation")
    line2, = ax2.plot(ygt[:,-1],label="Groundtruth")
    ax2.legend()
    t = np.array(t)
    t_ = np.squeeze(t)
    d = ax1.imshow(t_, cmap ='jet', interpolation ='nearest', origin ='lower')
    plt.colorbar(d, ax=ax1)
    plt.title(name)
    plt.show(block=False)
    plt.pause(1)  
