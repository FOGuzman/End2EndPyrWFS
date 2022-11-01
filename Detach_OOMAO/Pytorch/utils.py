import os
import datetime

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
 txt7 = "Constrain pupils     = {}\n".format(wfs.PupilConstrain)
 txt8 = "Readout noise        = {}\n".format(wfs.ReadoutNoise)
 txt9 = "Photon Noise         = {}\n".format(wfs.PhotonNoise)
 txt10= "# Photons Background = {}\n".format(wfs.nPhotonBackground)
 txt11= "Quantum Efficiency   = {}\n".format(wfs.quantumEfficiency)
 txt12= "Result path          = {}\n".format(result_path)
 txt13= "Loss function        = {}\n".format(loss)
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
 file_object.write(txt13)
 # Close the file
 file_object.close()   
 print("Log done!")      