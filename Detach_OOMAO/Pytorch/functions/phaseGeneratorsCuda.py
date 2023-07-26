import numpy as np
import math
from mpmath import *
import scipy.special as scp
import torch


def cart2pol(x, y):
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)
    return(rho, phi) 



def ZernikeProfile(batch,RandGain,wfs):
    coefs = (np.random.rand(len(wfs.jModes),batch)-0.5)*RandGain
    randomZernikeProfile = np.matmul(wfs.modes,coefs)
    phaseMap = np.reshape(randomZernikeProfile,(wfs.nPxPup,wfs.nPxPup,batch))
    phaseMap = np.moveaxis(phaseMap,2,0)
    
    return(phaseMap,coefs)




def freqspace(n):
    device = n.device  # Get the device of the input tensor n
    n = [n, n]
    a = torch.arange(0, n[1], 1,device=device)
    b = torch.arange(0, n[0], 1,device=device)
    f1 = (a - torch.floor(n[1] / 2)) * (2 / n[1])
    f2 = (b - torch.floor(n[0] / 2)) * (2 / n[0])
    f1, f2 = torch.meshgrid(f1, f2)
    return f1, f2




def spectrum(f,r0,L0,fR0):
    out = (24.*math.gamma(6/5)/5)**(5./6)*(math.gamma(11/6)**2/(2*math.pi**(11/3)))*r0**(-5/3)
    out = out*(f**2 + 1/L0**2)**(-11/6)
    out = fR0*out
    return(out)
  
  
  
  
def PerformRxRy(fx,fy,fc,nActuator,D,r0,L0,fR0,modulation,binning,noiseVariance):
    device = fx.device
    nL      = nActuator - 1
    d       = D/nL
    f       = torch.sqrt(torch.absolute(fx)**2+torch.absolute(fy)**2)
    Wn      = noiseVariance/(2*fc)**2
    Wphi    = spectrum(f,r0,L0,fR0)
    u       = fx
           
    umod = 1/(2*d)/(nL/2)*modulation
    Sx = torch.zeros(size=u.shape,dtype=torch.complex64,device=device)
    idx = torch.absolute(u) > umod
    
    Sx[idx] = 1j*torch.sign(u[idx])
    idx = torch.absolute(u) <= umod
 
    
    Av = torch.sinc(binning * d * u) * torch.transpose(torch.sinc(binning * d * u), 0, 1)
    Sy = torch.transpose(Sx, 0, 1)

    #reconstruction filter
    AvRec = Av
    SxAvRec = Sx*AvRec
    SyAvRec = Sy*AvRec
               
    # --------------------------------------
    #   MMSE filter
    # --------------------------------------
    gPSD = torch.absolute(Sx*AvRec)**2 + torch.absolute(Sy*AvRec)**2 + Wn/Wphi +1e-7
    Rx = torch.conj(SxAvRec)/gPSD
    Ry = torch.conj(SyAvRec)/gPSD
    return(Rx,Ry)
    
    


def sombrero(n,x):
    device = x.device
    if n==0:
       out = besselj(0,x)/x
    else:
       if n>1:
          out = torch.zeros(size=x.shape,device=device)
       else:
          out = 0.5*torch.ones(size=x.shape,device=device)
       u = x != 0
       x = x[u]
       aux = torch.from_numpy(scp.j1(x.cpu().numpy())).to(device)  #order 1 bessel
       out[u] = aux/x
       return(out)
     
     
def pistonFilter(D,f):
    red = math.pi*D*f
    sm = sombrero(1,red)
    out = 1 - 4*sm**2
    return(out)


def fittingPSD(fx,fy,fc,shape,nTimes,r0,L0,fR0,D):
    device=fx.device
    fx,fy = freqspace(fx.shape[1]*nTimes)
    fx = fx*fc*nTimes
    fy = fy*fc*nTimes
    out = torch.zeros(size=fx.shape,device=device)
    #cv2_imshow(np.logical_or((np.absolute(fx)>fc),(np.absolute(fy)>fc))*255)
    if shape == "square":
          index  = torch.logical_or((torch.absolute(fx)>fc),(torch.absolute(fy)>fc))        
    else:
          index  = torch.sqrt(torch.absolute(fx)**2+torch.absolute(fy)**2) > fc

    f_v     = torch.sqrt(torch.absolute(fx[index])**2+torch.absolute(fy[index])**2)
    out[index] = spectrum(f_v,r0,L0,fR0)
    out = out*pistonFilter(D,torch.sqrt(torch.absolute(fx)**2+torch.absolute(fy)**2))
    return(out)
    
def noisePSD(fx,fy,fc,Rx,Ry,noiseVariance,D):
    device = fx.device
    out   = torch.zeros(size=(fx.shape),device=device)
    if noiseVariance>0:
       index = torch.logical_not(torch.logical_and(torch.logical_or((torch.absolute(fx)>fc),(torch.absolute(fy)>fc)),torch.sqrt(torch.absolute(fx)**2+torch.absolute(fy)**2)>0))
       out[index] = noiseVariance/(2*fc)**2*(torch.absolute(Rx[index])**2 + torch.absolute(Ry[index])**2)
       out = out*pistonFilter(D,torch.sqrt(torch.absolute(fx)**2+torch.absolute(fy)**2))
       return(out)
       
def SxyAv(fx,fy,D,nLenslet):
    d = D/(nLenslet)
    Sx = 1j*2*math.pi*fx*d
    Sy = 1j*2*math.pi*fy*d
    Av = torch.sinc(d*fx)*torch.sinc(d*fy)*torch.exp(1j*math.pi*d*(fx+fy))
    SxAv = Sx*Av
    SyAv = Sy*Av
    return(SxAv,SyAv)

def anisoServoLagPSD(fx,fy,fc,r0,L0,fR0,D):
    device = fx.device
    out   = torch.zeros(size=fx.shape,device=device)
    index = torch.logical_not(torch.logical_or((torch.absolute(fx)>=fc),(torch.absolute(fy)>=fc)))
    pf = pistonFilter(D,torch.sqrt(torch.absolute(fx)**2+torch.absolute(fy)**2))
    fx     = fx[index]
    fy     = fy[index]
    rtf = 1
    out[index] =  rtf * spectrum(torch.sqrt(torch.absolute(fx)**2+torch.absolute(fy)**2),r0,L0,fR0)
    out = pf*out
    return(out)



def graduatedHeaviside(x,n):
    dx = x[0,1]-x[0,0]
    if dx == 0:
       dx = x[1,1]-x[1,0]
    out = np.zeros((np.size(x,0),np.size(x,1)))   
    out[-dx*n<=x] = 0.5
    out[x>dx*n] = 1
    return(out)


def GetTurbulenceParameters(wfs,resAO,nLenslet,r0,L0,fR0,noiseVariance,nTimes,n_lvl):
    device = resAO.device
    Samp = wfs.samp
    nPxPup = wfs.nPxPup
    modulation = wfs.modulation
    binning = 1
    D = wfs.D
    N             = torch.tensor(2*Samp*nPxPup).to(device)
    L             = (N-1)*D/(nPxPup-1)
    fc            = 1/binning*0.5*(nLenslet)/D
    
    fx,fy = freqspace(resAO)
    fx = fx*fc + 1e-7
    fy = fy*fc + 1e-7
    Rx, Ry    = PerformRxRy(fx,fy,fc,nLenslet+1,D,r0,L0,fR0,modulation,binning,noiseVariance)
    psdFit    = fittingPSD(fx,fy,fc,"square",nTimes,r0,L0,fR0,D)/r0**(-5/3)
    psdNoise  = noisePSD(fx,fy,fc,Rx,Ry,noiseVariance,D)/noiseVariance
    
    fxExt,fyExt = freqspace(fx.shape[1]*nTimes)
    fxExt = fxExt*fc*nTimes
    fyExt = fyExt*fc*nTimes
    index = torch.logical_and(torch.absolute(fxExt)<fc,torch.absolute(fyExt)<fc)
       
    psdAO_mean = torch.zeros(size=fxExt.shape,device=device)
    aSlPSD = anisoServoLagPSD(fx,fy,fc,r0,L0,fR0,D)
    psdFact = aSlPSD  + psdNoise*n_lvl
    psdAO_mean[index] = psdFact.flatten()
    psdAO_mean = psdAO_mean + psdFit*r0**(-5/3)
    
    fourierSampling = 1/L
    
    fx,fy = freqspace(N)
    fr,a  = cart2pol(fx,fy)
    fr    = torch.fft.fftshift(fr*(N-1)/L/2)
    #idx = torch.nonzero(fr.flatten() == 0)
    idx = L/L
    atm = {
   "psdAO_mean": psdAO_mean,
   "N": N,
   "fourierSampling": fourierSampling,
   "idx": idx,
   "pupil": wfs.pupil,
   "nPxPup": nPxPup
    }
    return(atm)

def GetPhaseMapAndZernike(atm,CM,batch_size):
    psdAO_mean = atm['psdAO_mean']
    device = psdAO_mean.device
    N = atm['N']
    fourierSampling = atm['fourierSampling']
    idx = atm['idx']
    pupil = atm['pupil']
    nPxPup = torch.tensor(atm['nPxPup']).to(device)

    rngTensor = torch.randn(batch_size, 1, N, N,device=device)
    
    phaseMap = torch.fft.ifft2(idx*torch.sqrt(torch.fft.fftshift(psdAO_mean))*torch.fft.fft2(rngTensor)/N)*fourierSampling
    phaseMap = torch.real(phaseMap)*N**2
    phaseMap = pupil*phaseMap[:,:,0:nPxPup,0:nPxPup]
    phaseMap_vec = phaseMap.view(phaseMap.shape[0], -1).t()
    Ze = torch.matmul(CM,phaseMap_vec)
    return(phaseMap,Ze)


def GetPhaseMapAndZernike_CUDA(psdAO_mean,N,fourierSampling,idx,pupil,nPxPup,CM):
    n = N.type(torch.int16)
    randMap = torch.cuda.FloatTensor(torch.Size((n,n)))
    randMap = torch.randn((n,n),out=randMap)
    phaseMap = torch.fft.ifft2(idx*torch.sqrt(torch.fft.fftshift(psdAO_mean))*torch.fft.fft2(randMap)/N)*fourierSampling
    phaseMap = phaseMap.real*N**2
    phaseMap = pupil*phaseMap[0:nPxPup,0:nPxPup]
    Ze = torch.matmul(CM,torch.reshape(phaseMap,[-1]))
    return(phaseMap,Ze)



def GenerateFourierPhase(pupil,resAO,nLenslet,D,r0,L0,fR0,modulation,binning,noiseVariance,fc,nTimes,n_lvl,L,N,nPxPup):
    fx,fy = freqspace(resAO)
    fx = fx*fc + 1e-7
    fy = fy*fc + 1e-7
    Rx, Ry    = PerformRxRy(fx,fy,fc,nLenslet+1,D,r0,L0,fR0,modulation,binning,noiseVariance)
    psdFit    = fittingPSD(fx,fy,fc,"square",nTimes,r0,L0,fR0,D)/r0**(-5/3)
    psdNoise  = noisePSD(fx,fy,fc,Rx,Ry,noiseVariance,D)/noiseVariance
    
    fxExt,fyExt = freqspace(np.size(fx,1)*nTimes)
    fxExt = fxExt*fc*nTimes
    fyExt = fyExt*fc*nTimes
    index = np.logical_and(np.absolute(fxExt)<fc,np.absolute(fyExt)<fc)
    
    SxAv,SyAv = SxyAv(fx,fy,D,nLenslet)
    
    psdAO_mean = np.zeros((np.size(fxExt,0),np.size(fxExt,1)))
    aSlPSD = anisoServoLagPSD(fx,fy,fc,r0,L0,fR0,D)
    psdFact = aSlPSD  + psdNoise*np.mean(n_lvl)
    psdAO_mean[index] = psdFact.flatten()
    psdAO_mean = psdAO_mean + psdFit*r0**(-5/3)
    
    fourierSampling = 1/L
    
    fx,fy = freqspace(N)
    fr,a  = cart2pol(fx,fy)
    fr    = np.fft.fftshift(fr*(N-1)/L/2)
    idx = np.where(fr.flatten() == 0)
    idx = idx[0][0]+1
    N = np.int16(N)
    nPxPup = np.int32(nPxPup)
    phaseMap = np.fft.ifft2(idx*torch.sqrt(torch.fft.fftshift(psdAO_mean))*torch.fft.fft2(np.random.randn(N,N))/N)*fourierSampling
    phaseMap = phaseMap.real*N**2
    phaseMap = pupil*phaseMap[0:nPxPup,0:nPxPup]
    return(phaseMap)  


def GenerateFourierPhaseXY(r0,L0,D,resAO,nLenslet,nTimes,n_lvl,noiseVariance,CM,wfs):
    fR0           = 1
    binning       = 1
    Samp          = wfs.samp
    nPxPup        = wfs.nPxPup
    nTimes        = wfs.fovInPixel/resAO

    N             = 2*Samp*nPxPup;
    L             = (N-1)*D/(nPxPup-1)
    fc            = 1/binning*0.5*(nLenslet)/D
    
    phaseMap = GenerateFourierPhase(wfs.pupil,resAO,nLenslet,D,r0,L0,fR0,wfs.modulation,binning,noiseVariance,fc,nTimes,n_lvl,L,N,nPxPup)

    Ze = np.matmul(CM,torch.reshape(torch.tensor(phaseMap),[-1]))
    return(phaseMap,Ze)