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

y1 = angle(exp(1j.*pyrMask).*exp(1j.*fftshift(OL1_base)));
y2 = angle(exp(1j.*fftshift(OL1_base)));
y3 = angle(exp(1j.*pyrMask).*exp(1j.*fftshift(OL1_noise)));
y4 = angle(exp(1j.*fftshift(OL1_noise)));

%%
UW = @(x) unwrap(unwrap(x,[],1),[],2);


fig = figure('Color','w','Position',[420 345 1471 627]);
ha = tight_subplot(1,4,[.0 .0],[.01 .01],[.01 .06]);



axes(ha(1));
imagesc(y1,[-pi pi]);colormap jet;colorbar;axis image;axis off
axes(ha(2));
imagesc(y2,[-pi pi]);colormap jet;colorbar;axis image;axis off
axes(ha(3));
imagesc(y3,[-pi pi]);colormap jet;colorbar;axis image;axis off
axes(ha(4));
imagesc(y4,[-pi pi]);colormap jet;cb3 = colorbar;axis image;axis off


cb3.FontSize = 22;
cb3.TickLabelInterpreter = 'latex';
cb3.Ticks = [-pi pi];
cb3.TickLabels = {'$-\pi$';'$\pi$'};
%%
fold = "./figures/";
name = "de_examples.pdf";
exportgraphics(fig,fold+name)