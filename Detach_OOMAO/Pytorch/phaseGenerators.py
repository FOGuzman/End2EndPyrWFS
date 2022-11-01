import numpy as np
import math
from mpmath import *
import scipy.special as scp
import torch


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi) 



def ZernikeProfile(batch,RandGain,wfs):
    coefs = (np.random.rand(len(wfs.jModes),batch)-0.5)*RandGain
    randomZernikeProfile = np.matmul(wfs.modes,coefs)
    phaseMap = np.reshape(randomZernikeProfile,(wfs.nPxPup,wfs.nPxPup,batch))
    phaseMap = np.moveaxis(phaseMap,2,0)
    
    return(phaseMap,coefs)




def freqspace(n,flag):
    n = [n,n]
    a = np.arange(0,n[1],1)
    b = np.arange(0,n[0],1)
    f1 = (a-np.floor(n[1]/2))*(2/(n[1]))
    f2 = (b-np.floor(n[0]/2))*(2/(n[0]))
    f1,f2 = np.meshgrid(f1,f2)
    return(f1,f2)




def spectrum(f,r0,L0,fR0):
    out = (24.*math.gamma(6/5)/5)**(5./6)*(math.gamma(11/6)**2/(2*math.pi**(11/3)))*r0**(-5/3)
    out = out*(f**2 + 1/L0**2)**(-11/6)
    out = fR0*out
    return(out)
  
  
  
  
def PerformRxRy(fx,fy,fc,nActuator,D,r0,L0,fR0,modulation,binning,noiseVariance):
    nL      = nActuator - 1
    d       = D/nL
    f       = np.sqrt(np.absolute(fx)**2+np.absolute(fy)**2)
    Wn      = noiseVariance/(2*fc)**2
    Wphi    = spectrum(f,r0,L0,fR0)
    u       = fx;
           
    umod = 1/(2*d)/(nL/2)*modulation
    Sx = np.zeros((np.size(u,0),np.size(u,1)),dtype=complex)
    idx = np.absolute(u) > umod
    
    Sx[idx] = 1j*np.sign(u[idx])
    idx = np.absolute(u) <= umod
 
    Sx[idx] = 2*1j/math.pi*np.arcsin(u[idx]/umod)
    
    Av = np.sinc(binning*d*u)*np.transpose(np.sinc(binning*d*u))
    Sy = np.transpose(Sx)
    SxAv = Sx*Av
    SyAv = Sy*Av

    #reconstruction filter
    AvRec = Av
    SxAvRec = Sx*AvRec
    SyAvRec = Sy*AvRec
               
    # --------------------------------------
    #   MMSE filter
    # --------------------------------------
    gPSD = np.absolute(Sx*AvRec)**2 + np.absolute(Sy*AvRec)**2 + Wn/Wphi +1e-7
    Rx = np.conj(SxAvRec)/gPSD;
    Ry = np.conj(SyAvRec)/gPSD;
    return(Rx,Ry)
    
    


def sombrero(n,x):
    if n==0:
       out = besselj(0,x)/x
    else:
       if n>1:
          out = np.zeros((np.size(x,0),np.size(x,1)))
       else:
          out = 0.5*np.ones((np.size(x,0),np.size(x,1)))
       u = x != 0
       x = x[u]
       aux = scp.j1(x)  #order 1 bessel
       out[u] = aux/x
       return(out)
     
     
def pistonFilter(D,f):
    red = math.pi*D*f;
    sm = sombrero(1,red)
    out = 1 - 4*sm**2
    return(out)


def fittingPSD(fx,fy,fc,shape,nTimes,r0,L0,fR0,D):
    fx,fy = freqspace(np.size(fx,1)*nTimes,"meshgrid")
    fx = fx*fc*nTimes
    fy = fy*fc*nTimes
    out = np.zeros((np.size(fx,0),np.size(fx,1)))
    #cv2_imshow(np.logical_or((np.absolute(fx)>fc),(np.absolute(fy)>fc))*255)
    if shape == "square":
          index  = np.logical_or((np.absolute(fx)>fc),(np.absolute(fy)>fc))        
    else:
          index  = np.sqrt(np.absolute(fx)**2+np.absolute(fy)**2) > fc

    f     = np.sqrt(np.absolute(fx[index])**2+np.absolute(fy[index])**2)
    out[index] = spectrum(f,r0,L0,fR0)
    out = out*pistonFilter(D,np.sqrt(np.absolute(fx)**2+np.absolute(fy)**2))
    return(out)
    
def noisePSD(fx,fy,fc,Rx,Ry,noiseVariance,D):
    out   = np.zeros((np.size(fx,0),np.size(fx,1)))
    if noiseVariance>0:
       index = np.logical_not(np.logical_and(np.logical_or((np.absolute(fx)>fc),(np.absolute(fy)>fc)),np.sqrt(np.absolute(fx)**2+np.absolute(fy)**2)>0))
       out[index] = noiseVariance/(2*fc)**2*(np.absolute(Rx[index])**2 + np.absolute(Ry[index])**2)
       out = out*pistonFilter(D,np.sqrt(np.absolute(fx)**2+np.absolute(fy)**2))
       return(out)
       
def SxyAv(fx,fy,D,nLenslet):
    d = D/(nLenslet);
    Sx = 1j*2*math.pi*fx*d ;
    Sy = 1j*2*math.pi*fy*d ;
    Av = np.sinc(d*fx)*np.sinc(d*fy)*np.exp(1j*math.pi*d*(fx+fy))
    SxAv = Sx*Av
    SyAv = Sy*Av
    return(SxAv,SyAv)

def anisoServoLagPSD(fx,fy,fc,Rx,Ry,SxAv,SyAv,r0,L0,fR0,D):
    out   = np.zeros((np.size(fx,0),np.size(fx,1)))
    index = np.logical_not(np.logical_or((np.absolute(fx)>=fc),(np.absolute(fy)>=fc)))
    pf = pistonFilter(D,np.sqrt(np.absolute(fx)**2+np.absolute(fy)**2))
    fx     = fx[index]
    fy     = fy[index]
    rtf = 1
    out[index] =  rtf * spectrum(np.sqrt(np.absolute(fx)**2+np.absolute(fy)**2),r0,L0,fR0)
    out = pf*out.real
    return(out)



def graduatedHeaviside(x,n):
    dx = x[0,1]-x[0,0]
    if dx == 0:
       dx = x[1,1]-x[1,0]
    out = np.zeros((np.size(x,0),np.size(x,1)))   
    out[-dx*n<=x] = 0.5
    out[x>dx*n] = 1
    return(out)








def GenerateFourierPhase(pupil,resAO,nLenslet,D,r0,L0,fR0,modulation,binning,noiseVariance,fc,nTimes,n_lvl,L,N,nPxPup):
    fx,fy = freqspace(resAO,"meshgrid")
    fx = fx*fc + 1e-7
    fy = fy*fc + 1e-7
    Rx, Ry    = PerformRxRy(fx,fy,fc,nLenslet+1,D,r0,L0,fR0,modulation,binning,noiseVariance)
    psdFit    = fittingPSD(fx,fy,fc,"square",nTimes,r0,L0,fR0,D)/r0**(5/3)
    psdNoise  = noisePSD(fx,fy,fc,Rx,Ry,noiseVariance,D)/noiseVariance
    
    fxExt,fyExt = freqspace(np.size(fx,1)*nTimes,"meshgrid")
    fxExt = fxExt*fc*nTimes;
    fyExt = fyExt*fc*nTimes;
    index = np.logical_and(np.absolute(fxExt)<fc,np.absolute(fyExt)<fc)
    
    SxAv,SyAv = SxyAv(fx,fy,D,nLenslet)
    
    psdAO_mean = np.zeros((np.size(fxExt,0),np.size(fxExt,1)))
    aSlPSD = anisoServoLagPSD(fx,fy,fc,Rx,Ry,SxAv,SyAv,r0,L0,fR0,D)
    psdFact = aSlPSD  + psdNoise*np.mean(n_lvl)
    psdAO_mean[index] = psdFact.flatten()
    psdAO_mean = psdAO_mean + psdFit*r0**(-5/3)
    
    fourierSampling = 1/L;
    
    fx,fy = freqspace(N,"meshgrid")
    fr,a  = cart2pol(fx,fy)
    fr    = np.fft.fftshift(fr*(N-1)/L/2)
    N = np.int16(N)
    nPxPup = np.int32(nPxPup)
    phaseMap = np.fft.ifft2(np.sqrt(np.fft.fftshift(psdAO_mean))*np.fft.fft2(np.random.randn(N,N))/N)*fourierSampling
    phaseMap = phaseMap.real*N**2
    phaseMap = pupil*phaseMap[0:nPxPup,0:nPxPup]
    return(phaseMap)  


def GenerateFourierPhaseXY(r0,L0,D,resAO,nLenslet,nTimes,n_lvl,noiseVariance,wfs):
    fR0           = 1
    binning       = 1
    Samp          = wfs.samp
    nPxPup        = wfs.nPxPup
    nTimes        = wfs.fovInPixel/resAO

    N             = 2*Samp*nPxPup;
    L             = (N-1)*D/(nPxPup-1)
    fc            = 1/binning*0.5*(nLenslet)/D
    
    phaseMap = GenerateFourierPhase(wfs.pupil,resAO,nLenslet,D,r0,L0,fR0,wfs.modulation,binning,noiseVariance,fc,nTimes,n_lvl,L,N,nPxPup)
    IM = None
    #%% Calibration layer
    for k in range(len(wfs.jModes)):
        imMode = torch.reshape(torch.tensor(wfs.modes[:,k]),(nPxPup,nPxPup))
        Zv = torch.unsqueeze(torch.reshape(imMode,[-1]),1)
        if IM is not None:
                IM = torch.concat([IM,Zv],1)
        else:
                IM = Zv
    
    
    CM = torch.linalg.pinv(IM)
    Ze = np.matmul(CM,torch.reshape(torch.tensor(phaseMap),[-1]))
    return(phaseMap,Ze)