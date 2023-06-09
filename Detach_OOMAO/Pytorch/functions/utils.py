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
 txt1 = "Modulation           = {}\n".format(wfs.modulation)
 txt2 = "Sampling factor      = {}\n".format(wfs.samp)
 txt3 = "Sensor resolution    = {}\n".format(wfs.nPxPup)
 txt4 = "Pyramid rooftop      = {}\n".format(wfs.rooftop)
 txt5 = "Pyramid alpha        = {}\n".format(wfs.alpha)
 txt6 = "Zernikes used        = {}\n".format(wfs.zModes)
 txt7 = "Readout noise        = {}\n".format(wfs.ReadoutNoise)
 txt8 = "Photon Noise         = {}\n".format(wfs.PhotonNoise)
 txt9 = "# Photons Background = {}\n".format(wfs.nPhotonBackground)
 txt10= "Quantum Efficiency   = {}\n".format(wfs.quantumEfficiency)
 txt11= "Result path          = {}\n".format(result_path)
 txt12= "Loss function        = {}\n".format(loss)
 file_object.write(txt1)
 file_object.write(txt2)
 file_object.write(txt3)
 file_object.write(txt4)
 file_object.write(txt5)
 file_object.write(txt6)
 file_object.write(txt7)
 file_object.write(txt8)
 file_object.write(txt9)
 file_object.write(txt10)
 file_object.write(txt11)
 file_object.write(txt12)
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
