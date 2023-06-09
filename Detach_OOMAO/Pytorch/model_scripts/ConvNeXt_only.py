import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
import numpy as np
from functions.oomao_functions import *
from functions.phaseGenerators import *
from functions.Propagators import *
from torch import unsqueeze as UNZ

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            #return nn.LayerNorm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

class OptimizedPyramid(nn.Module):
    def __init__(self, wfs):
        super().__init__()
        self.batchSize = wfs.batchSize
        self.nPxPup = wfs.nPxPup
        self.modulation = wfs.modulation
        self.samp = wfs.samp
        self.rooftop = wfs.rooftop
        self.alpha = wfs.alpha
        self.jModes = wfs.jModes
        self.amplitude = wfs.amplitude
        self.ReadoutNoise = wfs.ReadoutNoise
        self.PhotonNoise = torch.tensor(wfs.PhotonNoise)
        self.quantumEfficiency = torch.tensor(wfs.quantumEfficiency)
        self.nPhotonBackground = torch.tensor(wfs.nPhotonBackground)
        if wfs.modulation > 0:
            self.ModPhasor = torch.permute(torch.tensor(wfs.ModPhasor),(2,0,1))
        self.fovInPixel    = torch.tensor(wfs.fovInPixel)
        self.pupil = torch.tensor(wfs.pupil)
        self.pyrMask = torch.tensor(wfs.pyrMask,dtype=torch.complex64)
        self.modes = torch.tensor(wfs.modes)       
        self.BatchModes = torch.zeros(size=(len(self.jModes),1,self.nPxPup,self.nPxPup)).cuda()
        for k in range(len(self.jModes)):           
            zim = torch.reshape(self.modes[:,k],(self.nPxPup,self.nPxPup))
            zim = torch.unsqueeze(zim,0).cuda()
            self.BatchModes[k,:,:,:] = zim

        self.pupilLogical = torch.tensor(wfs.pupilLogical)
        self.Flat = torch.ones((self.nPxPup,self.nPxPup))*self.pupilLogical
        self.Flat = UNZ(UNZ(self.Flat,0),0).cuda()

        self.NN = ConvNeXt(in_chans=1, num_classes=len(self.jModes),depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        self.NN = self.NN.cuda()
        
        ## CUDA
        if torch.cuda.is_available() == 1:
            self.pyrMask = self.pyrMask.cuda()
            self.pupil = self.pupil.cuda()
            if wfs.modulation > 0:
                self.ModPhasor = self.ModPhasor.cuda()
        self.PhotonNoise = self.PhotonNoise.cuda()
        self.quantumEfficiency = self.quantumEfficiency.cuda()
        self.nPhotonBackground = self.nPhotonBackground.cuda()    
            
            

    def forward(self, inputs):
        #propagation of X
        Ip = Prop2VanillaPyrWFS_torch(inputs,self)
        #Photon noise
        if self.PhotonNoise == 1:
            Ip = AddPhotonNoise(Ip,self)          
        #Read out noise 
        if self.ReadoutNoise != 0:
            Ip = Ip + torch.normal(0,self.ReadoutNoise,size=Ip.shape).cuda()   
        
        # Normalization
        Inorm = torch.sum(torch.sum(torch.sum(Ip,-1),-1),-1)
        Ip = Ip/UNZ(UNZ(UNZ(Inorm,-1),-1),-1)
        # Estimation
        y = self.NN(Ip).permute(1,0)
        return y
    
       
       
            
class PyrModel(nn.Module):
    def __init__(self,wfs):
        super(PyrModel,self).__init__()
        self.prop = OptimizedPyramid(wfs)

    def forward(self, x):
        return self.prop(x)   