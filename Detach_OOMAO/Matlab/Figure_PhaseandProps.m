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

%mask1 = pyrMask+fftshift(OL1_base);
% [NetCM_base,NetI_0_base] = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,Samp...
%     ,modulation,rooftop,alpha,pupil,OL1_base,1);
% 
% [NetCM_noise,NetI_0_noise] = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,Samp...
%     ,modulation,rooftop,alpha,pupil,OL1_noise,1);

[PyrCM,PyrI_0,PyrIM] = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,Samp...
    ,modulation,rooftop,alpha,pupil,OL1_base,0);
        
%% Meas
r0            = 0.1;
ReadoutNoise = 0.5;
PhotonNoise = 1;
nPhotonBackground = 0.5;
quantumEfficiency = 1;
atm = GenerateAtmosphereParameters(nLenslet,D,binning,r0,L0,fR0,modulation,fovInPixel,resAO,Samp,nPxPup,pupil);

[x,Zg] = ComputePhaseScreen(atm,PhaseCM);
% [x,Zg] = GenerateFourierPhase(nLenslet,D,binning,r0,L0,fR0,modulation,...
%     fovInPixel,resAO,Samp,nPxPup,pupil,jModes,modes,PhaseCM);

y = PropagatePyr(fovInPixel,x,Samp,modulation,rooftop,alpha,pupil,nPxPup,OL1_base,1);
y_base = y/sum(y(:));
if PhotonNoise
y = AddPhotonNoise(y,nPhotonBackground,quantumEfficiency);
end
y = y +randn(size(y)).*ReadoutNoise;
Net_y_base = y/sum(y(:));

y = PropagatePyr(fovInPixel,x,Samp,modulation,rooftop,alpha,pupil,nPxPup,OL1_noise,1);
y_noise = y/sum(y(:));
if PhotonNoise
y = AddPhotonNoise(y,nPhotonBackground,quantumEfficiency);
end
y = y +randn(size(y)).*ReadoutNoise;
Net_y_noise = y/sum(y(:));

y = PropagatePyr(fovInPixel,x,Samp,modulation,rooftop,alpha,pupil,nPxPup,OL1_base,0);
if PhotonNoise
y = AddPhotonNoise(y,nPhotonBackground,quantumEfficiency);
end
y = y +randn(size(y)).*ReadoutNoise;
Pyr_y = y/sum(y(:));



%%
imagesc(PyrI_0)
mmax = max(max([y_base(:) Net_y_base(:) y_noise(:) Net_y_noise(:)]));
mmin = min(min([y_base(:) Net_y_base(:) y_noise(:) Net_y_noise(:)]));
dmax = max(max([OL1_base(:) OL1_noise(:)]));
dmin = min(min([OL1_base(:) OL1_noise(:)]));

fig = figure('Color','w','Position',[420 345 1471 627]);
ha = tight_subplot(2,4,[.0 .0],[.01 .01],[.01 .06]);

axes(ha(2));
imshow(255);
axes(ha(4));
imshow(255)

axes(ha(2));
imshow(ones(3,3,3));
axes(ha(4));
imshow(ones(3,3,3));
axes(ha(1));
imagesc(fftshift(OL1_base),[dmin dmax]);colormap jet;cb1 = colorbar;axis image;axis off
axes(ha(3));
imagesc(fftshift(OL1_noise),[dmin dmax]);colormap jet;cb2 = colorbar;axis image;axis off
axes(ha(5));
imagesc(y_base,[0 mmax]);colormap jet;colorbar;axis image;axis off
axes(ha(6));
imagesc(Net_y_base,[0 mmax]);colormap jet;colorbar;axis image;axis off
axes(ha(7));
imagesc(y_noise,[0 mmax]);colormap jet;colorbar;axis image;axis off
axes(ha(8));
imagesc(Net_y_noise,[0 mmax]);colormap jet;cb3 = colorbar;axis image;axis off


cb1.FontSize = 22;
cb1.TickLabelInterpreter = 'latex';
cb2.FontSize = 22;
cb2.TickLabelInterpreter = 'latex';
cb3.FontSize = 22;
cb3.TickLabelInterpreter = 'latex';

%%
fold = "./figures/";
name = "Meas_examples.pdf";
%exportgraphics(fig,fold+name)



%%
