addpath functions
clear all;clc

preFold1 = "../Preconditioners/nocap/base/checkpoint/OL1_R128_M0_RMSE0.02807_Epoch_91.mat";
preFold2 = "../Preconditioners/nocap/mod/OL1_R64_M2_RMSE0.005704_Epoch_109.mat";

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

y1 = fftshift(OL1_base);
y2 = fftshift(OL1_base);
y3 = fftshift(OL1_noise);
y4 = fftshift(OL1_noise);

%%
UW = @(x) unwrap(unwrap(x,[],1),[],2);


fig = figure('Color','w','Position',[420 345 1471 627]);
ha = tight_subplot(1,4,[.0 .0],[.01 .01],[.01 .06]);



axes(ha(1));
imagesc(y1);colormap jet;colorbar;axis image;axis off
axes(ha(2));
imagesc(y2);colormap jet;colorbar;axis image;axis off
axes(ha(3));
imagesc(y3);colormap jet;colorbar;axis image;axis off
axes(ha(4));
imagesc(y4);colormap jet;cb3 = colorbar;axis image;axis off


cb3.FontSize = 22;
cb3.TickLabelInterpreter = 'latex';
%%
fold = "./figures/";
name = "de_examples2.pdf";
exportgraphics(fig,fold+name)