import numpy as np
import matplotlib.pyplot as plt
import math
from mpmath import *
import scipy.special as scp
import torch
import torchvision.transforms.functional as F

def plot_ctensor(t):
    t = np.array(t)
    t_ = np.squeeze(np.angle(t))
    d = plt.imshow(t_, cmap ='jet', interpolation ='nearest', origin ='lower')
    plt.colorbar(d)
    plt.show(block=False)
    
def plot_tensor(t):
    t = np.array(t)
    t_ = np.squeeze(t)
    d = plt.imshow(t_, cmap ='jet', interpolation ='nearest', origin ='lower')
    plt.colorbar(d)
    plt.show(block=False)
     

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def cart2polCuda(x, y):
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)
    return(rho, phi)     
    
    
def CreateTelescopePupil(Npx,shapetype):
     x = np.arange(-(Npx-1)/2,(Npx-1)/2+1,1)  # change to +1 if torch of tf
     x = x/np.max(x)
     u = x
     v = x
            
     x,y = np.meshgrid(u,v)
     r,o = cart2pol(x,y)      

     if shapetype == "disc":
         out = r <= 1
         #print(out)
     if shapetype == "square":
         out = np.absolute(x)<=1 & np.absolute(y)<=1 
     if shapetype == "hex":
         out = np.absolute(x)<=np.sqrt(3)/2 & np.absolute(y)<=x/np.sqrt(3)+1 & np.absolute(y)<=-x/np.sqrt(3)+1

     return(out)
     
     
def freqspace(n,flag):
    n = [n,n]
    a = np.arange(0,n[1],1)
    b = np.arange(0,n[0],1)
    f1 = (a-np.floor(n[1]/2))*(2/(n[1]))
    f2 = (b-np.floor(n[0]/2))*(2/(n[0]))
    f1,f2 = np.meshgrid(f1,f2)
    return(f1,f2)


def graduatedHeaviside(x,n):
    dx = x[0,1]-x[0,0]
    if dx == 0:
       dx = x[1,1]-x[1,0]
    out = np.zeros((np.size(x,0),np.size(x,1)))   
    out[-dx*n<=x] = 0.5
    out[x>dx*n] = 1
    return(out)

def graduated_heaviside(x, n):
    dx = x[0, 1] - x[0, 0]
    if dx == 0:
        dx = x[1, 0] - x[0, 0]
    out = np.zeros_like(x)
    out[np.logical_and(-dx * n <= x, x <= dx * n)] = 0.5
    out[x > dx * n] = 1
    return out    


def createPyrMask(wfs):
    pixSide = wfs.fovInPixel
    rooftop = wfs.rooftop
    alpha   = wfs.alpha
    realAlpha = np.ones((1,4))*alpha
    imagAlpha = realAlpha.imag
    realAlpha = realAlpha.real
    nx = rooftop[0]
    ny = rooftop[1]
    fx,fy = freqspace(pixSide,"meshgrid")
    fx = fx*np.floor(pixSide/2)
    fy = fy*np.floor(pixSide/2)    
    #c1
    mask  = graduatedHeaviside(fx,nx)*graduatedHeaviside(fy,nx)
    phase = -realAlpha[:,0]*(fx+fy) + -imagAlpha[:,0]*(-fx+fy)
    pym   = mask*np.exp(1j*phase)
    #c2
    mask  = graduatedHeaviside(fx,ny)*graduatedHeaviside(-fy,-ny)
    phase = -realAlpha[:,1]*(fx-fy) + -imagAlpha[:,1]*(fx+fy)
    pym   = pym + mask*np.exp(1j*phase)
    #c3
    mask  = graduatedHeaviside(-fx,-nx)*graduatedHeaviside(-fy,-nx)
    phase = realAlpha[:,2]*(fx+fy) + -imagAlpha[:,2]*(fx-fy)
    pym   = pym + mask*np.exp(1j*phase)
    #c4
    mask  = graduatedHeaviside(-fx,-ny)*graduatedHeaviside(fy,ny)
    phase = -realAlpha[:,3]*(-fx+fy) + -imagAlpha[:,3]*(-fx-fy)
    pym   = pym + mask*np.exp(1j*phase)

    pyrMask   = np.fft.fftshift(pym/np.sum(np.abs(pym.flatten())))
    return(pyrMask)     



def CreateZernikePolynomials(wfs):
    nPxPup = wfs.nPxPup
    jModes = wfs.jModes
    pupilLogical = wfs.pupilLogical
    u = nPxPup
    u = np.linspace(-1,1,u)
    v = u
    x,y = np.meshgrid(u,v)
    r,o = cart2pol(x,y) 
    
    mode = jModes
    nf = [0]*len(mode)
    mf = [0]*len(mode)
    
    for cj in range(len(mode)):
        
        j = jModes[cj];
        n  = 0;
        m  = 0;
        j1 = j-1;
    
        while j1 > n:
            n  = n + 1
            j1 = j1 - n
            m  = (-1)**j * (n%2 + 2*np.floor((j1+(n+1)%2)/2))
        nf[cj] = np.int16(n)
        mf[cj] = np.int16(np.abs(m))
        
        
    nv = np.array(nf)
    mv = np.array(mf)
    nf  = len(jModes)
    fun = np.zeros((np.size(r),nf))
    r = np.transpose(r)
    o = np.transpose(o)
    r = r[pupilLogical]
    o = o[pupilLogical]
    pupilVec = pupilLogical.flatten()
    
    def R_fun(r,n,m):
        R=np.zeros(np.size(r))
        sran = int((n-m)/2)+1
        for s in range(sran):
            Rn = (-1)**s*np.prod(np.arange(1,(n-s)+1,dtype=float))*r**(n-2*s)
            Rd = (np.prod(np.arange(1,s+1))*np.prod(np.arange(1,((n+m)/2-s+1),dtype=float))*np.prod(np.arange(1,((n-m)/2-s)+1)))
            R = R + Rn/Rd
        return(R)    
    
    
    
    ind_m = list(np.array(np.nonzero(mv==0))[0])
    for cpt in ind_m:
    
        n = nv[cpt]
        m = mv[cpt]
        fun[pupilLogical.flatten(),cpt] = np.sqrt(n+1)*R_fun(r,n,m)
    
    mod_mode = jModes%2
    
    
    ind_m = list(np.array(np.nonzero(np.logical_and(mod_mode==0,mv!=0))))
    for cpt in ind_m:
        n = nv[cpt]
        m = mv[cpt]
        fun[pupilLogical.flatten(),cpt] = np.sqrt(n+1)*R_fun(r,n,m)*np.sqrt(2)*np.cos(m*o)
    
    
    ind_m = list(np.array(np.nonzero(np.logical_and(mod_mode==1,mv!=0))))
    for cpt in ind_m:
        n = nv[cpt]
        m = mv[cpt]
        fun[pupilLogical.flatten(),cpt] = np.sqrt(n+1)*R_fun(r,n,m)*np.sqrt(2)*np.sin(m*o)
        
        
    modes = fun
    return(modes)


def CreateModulationPhasor(wfs):
    x = np.arange(0,wfs.fovInPixel,1)/wfs.fovInPixel
    vv, uu = np.meshgrid(x,x)
    r,o = cart2pol(uu,vv)
    nTheta = np.round(2*math.pi*wfs.samp*wfs.modulation)
    ModPhasor = np.zeros((wfs.fovInPixel,wfs.fovInPixel,np.int16(nTheta)),dtype=complex)

    for kTheta in range(np.int16(nTheta)):

        theta = (kTheta)*2*math.pi/nTheta
        ph = math.pi*4*wfs.modulation*wfs.samp*r*np.cos(o+theta)
        fftPhasor = np.exp(-1j*ph)
        ModPhasor[:,:,kTheta] = fftPhasor
        
    return(ModPhasor)


def AddPhotonNoise(y,wfs):
    buffer    = y + wfs.nPhotonBackground
    y = y + torch.normal(0,1,size=y.shape).cuda()*(y + wfs.nPhotonBackground)
    index     = y<0
    y[index] = buffer[index]
    y = wfs.quantumEfficiency*y
    return y

def Propagate2Pyramid_torch(phaseMap,wfs):
    
    nTheta = np.round(2*math.pi*wfs.samp*wfs.modulation)
    nTheta = torch.tensor(nTheta)
    PyrQ  = torch.zeros((wfs.fovInPixel,wfs.fovInPixel))
    pyrMask = torch.unsqueeze(torch.tensor(wfs.pyrMask),0)
    pupil = torch.tensor(wfs.pupil)
    
    I4Q4 =  torch.zeros((wfs.fovInPixel,wfs.fovInPixel))
    pyrPupil = pupil*torch.exp(1j*phaseMap)
    subscale = 1/(2*wfs.samp)
    sx = torch.round(torch.tensor(wfs.fovInPixel*subscale)).to(torch.int16)   
    npv = ((wfs.fovInPixel-sx)/2).to(torch.int16)
    PyrQ = torch.unsqueeze(torch.nn.functional.pad(pyrPupil,(npv,npv,npv,npv), "constant", 0),1)
    
    if nTheta > 0:
        ModPhasor = torch.permute(wfs.ModPhasor,(2,0,1))
        buf = PyrQ*ModPhasor
        buf = torch.fft.fft2(buf)*pyrMask 
        buf = torch.fft.fft2(buf)      
        I4Q4 = torch.sum(torch.abs(buf)**2 ,1) 
        I4Q = I4Q4/nTheta
    else:
        buf = torch.fft.fft2(PyrQ)*pyrMask
        I4Q = torch.abs(torch.fft.fft2(buf))**2  
    I4Q = F.resize(I4Q,(sx,sx))*2*wfs.samp
    return(I4Q)



def Propagate2PyramidNoMod_torch(phaseMap,wfs):
    
    nTheta = np.round(2*math.pi*wfs.samp*wfs.modulation)
    nTheta = torch.tensor(nTheta)
    PyrQ  = torch.zeros((wfs.fovInPixel,wfs.fovInPixel))
    pyrMask = torch.unsqueeze(torch.tensor(wfs.pyrMask),0)
    pupil = torch.tensor(wfs.pupil)
    I4Q4 =  torch.zeros((wfs.fovInPixel,wfs.fovInPixel))
    pyrPupil = pupil*torch.exp(1j*phaseMap)
    subscale = 1/(2*wfs.samp)
    sx = torch.round(torch.tensor(wfs.fovInPixel*subscale)).to(torch.int16)   
    npv = ((wfs.fovInPixel-sx)/2).to(torch.int16)
    PyrQ = torch.unsqueeze(torch.nn.functional.pad(pyrPupil,(npv,npv,npv,npv), "constant", 0),1)
    
    buf = torch.fft.fft2(PyrQ)*pyrMask
    I4Q = torch.abs(torch.fft.fft2(buf))**2  
    I4Q = F.resize(I4Q,(sx,sx))*2*wfs.samp
    return(I4Q)



def Pro2OptPyrNoMod_torch(phaseMap,OL1,wfs):
    
    nTheta = np.round(2*math.pi*wfs.samp*wfs.modulation)
    nTheta = torch.tensor(nTheta)
    PyrQ  = torch.zeros((wfs.fovInPixel,wfs.fovInPixel))
    pyrMask = torch.unsqueeze(wfs.pyrMask,0)  
    pupil = wfs.pupil  
    pyrPupil = pupil*torch.exp(1j*phaseMap)
    subscale = 1/(2*wfs.samp)
    sx = torch.round(wfs.fovInPixel*subscale).to(torch.int16)   
    npv = ((wfs.fovInPixel-sx)/2).to(torch.int16)
    PyrQ = torch.nn.functional.pad(pyrPupil,(npv,npv,npv,npv), "constant", 0)
    buf = torch.fft.fft2(PyrQ)*pyrMask*OL1
    I4Q = torch.abs(torch.fft.fft2(buf))**2
    I4Q = F.resize(I4Q,(sx,sx))*2*wfs.samp
    return(I4Q)

def Pro2OptWFS_torch(phaseMap,OL1,wfs):
    
    nTheta = np.round(2*math.pi*wfs.samp*wfs.modulation)
    nTheta = torch.tensor(nTheta)
    PyrQ  = torch.zeros((wfs.fovInPixel,wfs.fovInPixel))
    yin = phaseMap
    pupil = wfs.pupil  
    I4Q4 =  torch.zeros((wfs.fovInPixel,wfs.fovInPixel))
    pyrPupil = pupil*torch.exp(1j*yin)
    subscale = 1/(2*wfs.samp)
    sx = torch.round(wfs.fovInPixel*subscale).to(torch.int16)   
    npv = ((wfs.fovInPixel-sx)/2).to(torch.int16)
    PyrQ = torch.unsqueeze(torch.nn.functional.pad(pyrPupil,(npv,npv,npv,npv), "constant", 0),1)
    if nTheta > 0:
        ModPhasor = torch.permute(wfs.ModPhasor,(2,0,1))
        buf = PyrQ*ModPhasor
        buf = torch.fft.fft2(buf)*OL1  
        buf = torch.fft.fft2(buf)      
        I4Q4 = torch.sum(torch.abs(buf)**2 ,1) 
        I4Q = I4Q4/nTheta
    else:
        buf = torch.fft.fft2(PyrQ)*OL1
        I4Q = torch.abs(torch.fft.fft2(buf))**2
    I4Q = F.resize(I4Q,(sx,sx))*2*wfs.samp
    return(I4Q)



