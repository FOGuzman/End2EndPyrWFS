addpath functions
clear all;clc

preFold1 = "../Preconditioners/nocap/base/checkpoint/OL1_R128_M0_RMSE0.02807_Epoch_91.mat";
preFold2 = "../Preconditioners/nocap/pnoise/checkpoint/OL1_R128_M0_RMSE0.05275_Epoch_118.mat";

binning       = 1;
D             = 8;
modulation    = 0;
nLenslet      = 16;
resAO         = 2*nLenslet+1;
L0            = 25;
fR0           = 1;
noiseVariance = 0.7;
n_lvl         = 0.1;             % noise level in rad^2
Samp          = 2;                % OVer-sampling factor
nPxPup        = 128;               % number of pixels to describe the pupil
alpha         = pi/2;
rooftop       = [0,0]; 
fovInPixel    = nPxPup*2*Samp;    % number of pixel to describe the PSD
PyrQ          = zeros(fovInPixel,fovInPixel);
I4Q4 = PyrQ;
nTimes        = fovInPixel/resAO;

N             = 2*Samp*nPxPup;
L             = (N-1)*D/(nPxPup-1);
pupil         = CreatePupil(nPxPup,"disc");

jModes = [2:12];

% Pyramid calibration
modes = CreateZernikePolynomials(nPxPup,jModes,pupil~=0);
flatMode = CreateZernikePolynomials(nPxPup,1,pupil~=0);


% Test
% Phase inversion
IM = [];     

for k = 1:length(jModes)

imMode = reshape(modes(:,k),[nPxPup nPxPup]);
IM = [IM imMode(:)];
end
PhaseCM = pinv(IM);

load(preFold1);OL1_base = OL1;
load(preFold2);OL1_noise = OL1;
%OL1_trained = 20*rand(512,512);
pyrMask = fftshift(angle(create_pyMask(fovInPixel,rooftop,alpha)));

[PyrCM,PyrI_0,PyrIM] = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,Samp...
    ,modulation,rooftop,alpha,pupil,OL1_base,0);
        
%% Meas
r0            = 0.1;
ReadoutNoise = 0;
PhotonNoise = 1;
nPhotonBackground = 0.0;
quantumEfficiency = 1;
atm = GenerateAtmosphereParameters(nLenslet,D,binning,r0,L0,fR0,modulation,fovInPixel,resAO,Samp,nPxPup,pupil);

[x,Zg] = ComputePhaseScreen(atm,PhaseCM);
% [x,Zg] = GenerateFourierPhase(nLenslet,D,binning,r0,L0,fR0,modulation,...
%     fovInPixel,resAO,Samp,nPxPup,pupil,jModes,modes,PhaseCM);


ReadoutNoise = 1;
PhotonNoise = 1;
nPhotonBackground = 0.1;
quantumEfficiency = 1;


y = PropagatePyr(fovInPixel,x,Samp,modulation,rooftop,alpha,pupil,nPxPup,OL1_base,0);
y_base = y/sum(y(:));
y1 = y_base;
if PhotonNoise
y = AddPhotonNoise(y,nPhotonBackground,quantumEfficiency);
end
y = y +randn(size(y)).*ReadoutNoise;
y2 = y/sum(y(:));


%


y = PropagatePyr(fovInPixel,x,Samp,modulation,rooftop,alpha,pupil,nPxPup,OL1_base,1);
y_base = y/sum(y(:));
y3 = y_base;
if PhotonNoise
y = AddPhotonNoise(y,nPhotonBackground,quantumEfficiency);
end
y = y +randn(size(y)).*ReadoutNoise;
y4 = y/sum(y(:));

%

%




%%
mmax = max(max([y1(:) y2(:) y3(:) y4(:)]));
mmin = min(min([y1(:) y2(:) y3(:) y4(:)]));

fig = figure('Color','w','Position',[415 340 1481 629]);
ha = tight_subplot(1,4,[.0 .0],[.01 .01],[.01 .06]);



axes(ha(1));
imagesc(y1,[0 mmax]);colormap jet;;axis image;axis off
axes(ha(2));
imagesc(y2,[0 mmax]);colormap jet;;axis image;axis off
axes(ha(3));
imagesc(y3,[0 mmax]);colormap jet;;axis image;axis off
axes(ha(4));
imagesc(y4,[0 mmax]);colormap jet;;axis image;axis off


%%
fold = "./figures/v3/";
name = "noise_examples.pdf";
exportgraphics(fig,fold+name)
