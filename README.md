# End2EndPyrWFS
Official implementation for Deep Optics Preconditioner for Enhanced Pyramid Wavefront Sensing

![ ](end2end_scheme.png)

# Installation
- install anaconda (https://www.anaconda.com/products/distribution)
- if windows: anaconda promprt
conda env create -f env.yml

# Description
This repository contains two main codes. The MATLAB code includes the basic functions extracted from [OOMAO](https://github.com/rconan/OOMAO), which are used for the analysis of results.
To train a diffractive element, we have also provided an implementation of the same equations in PyTorch.

#HowTo

- You must first generate data using the generators provided in the pytorch folder.

  Example for modulation 0, resolution 128 and get 35 Zernike decomposition of each phasemap:

  ```
  python DataGenerator.py --modulation 0 --samp 2 --D 8 --nPxPup 128 --zModes [2,36]

  ```

  or if Cuda is installed for GPU generation:

  ```
  python DataGeneratorCuda.py --modulation 0 --samp 2 --D 8 --nPxPup 128 --zModes [2,36]

  ```
  by default the script create 10000 training phases and zernike decompositions and 1000 for validation. 
  
- Then for training you have to use the same parameters generated:

  ```
  python E2E_main.py --modulation 0 --samp 2 --D 8 --nPxPup 128 --zModes [2,36] --batchSize 1

  ```
  
  check the script help for extra parameters like noise and pyramid shape
  
Once the training is finished, and for each epoch, a phase matrix of size nPxPup x samp, with the extension .mat is created. This file can be directly loaded into MATLAB.

# Reproducing Results
All the figures generated in the research paper were obtained using the scripts in the MATLAB folder. Please ensure that you set up the path to the Diffractive Element (saved as a .mat file) in the first lines of each script you wish to run. Here is a list of available figures that can be reproduced:

```
Figure4A_ComputePlot.m
Figure4B_ComputePlot.m
Figure5_ComputePlot.m
Figure6A_ComputePlot.m
Figure6B_ComputePlot.m
Figure7_ComputePlot.m
Figure8A_ComputePlot.m
Figure8B_ComputePlot.m
Figure9_ComputePlot.m
Figure10A_ComputePlot.m
Figure10B_ComputePlot.m
Figure11_ComputePlot.m

```


# Citation
If you find our project useful, please cite:

# Contact
You can contact Felipe Guzman by sending mail to felipe.guzman.v@mail.pucv.cl
