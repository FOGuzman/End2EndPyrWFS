%% Branch camilo Oct2022
addpath functions
clear all

preFold = "../Best_linear/OL1_R128_M0_RMSE0.0833_Epoch_84.mat";

binning       = 1;
D             = 8;
modulation    = 0;
nLenslet      = 16;
resAO         = 2*nLenslet+1;
r0            = 0.8;
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
jModes = [2:36];
ReadoutNoise = 0;


%% Pyramid calibration
modes = CreateZernikePolynomials(nPxPup,jModes,pupil~=0);
flatMode = CreateZernikePolynomials(nPxPup,1,pupil~=0);


%% Test
% Phase inversion
IM = [];     

for k = 1:length(jModes)

imMode = reshape(modes(:,k),[nPxPup nPxPup]);
IM = [IM imMode(:)];
end
PhaseCM = pinv(IM);

load(preFold);OL1_trained = OL1;
%OL1_trained = 20*rand(512,512);

[NetCM,NetI_0] = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,Samp...
    ,modulation,rooftop,alpha,pupil,OL1_trained,1);

[PyrCM,PyrI_0] = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,Samp...
    ,modulation,rooftop,alpha,pupil,OL1_trained,0);
        
%% Meas
r0            = 0.8;

[x,Zg] = GenerateFourierPhase(nLenslet,D,binning,r0,L0,fR0,modulation,...
    fovInPixel,resAO,Samp,nPxPup,pupil,jModes,modes,PhaseCM);

y = PropagatePyr(fovInPixel,x,Samp,modulation,rooftop,alpha,pupil,nPxPup,OL1_trained,1);
y = y +randn(size(y)).*ReadoutNoise;
Net_y = y/sum(y(:))-NetI_0;
% Estimation
NetZe = NetCM*Net_y(:);

y = PropagatePyr(fovInPixel,x,Samp,modulation,rooftop,alpha,pupil,nPxPup,OL1_trained,0);
y = y +randn(size(y)).*ReadoutNoise;
Pyr_y = y/sum(y(:))-PyrI_0;
% Estimation
PyrZe = PyrCM*Pyr_y(:);


imsc = [min(x(:)),max(x(:))];
figure('Position',[8 81 1886 874]),
subplot(341)
imagesc(x,[imsc(1) imsc(2)]);axis image;colorbar
title('Input Phase')
subplot(342)
imagesc(reshape(modes*Zg,[nPxPup nPxPup]));axis image;colorbar
title('Phase inversion')
subplot(343)
imagesc(reshape(modes*PyrZe,[nPxPup nPxPup]));axis image;colorbar
title('Pyramid estimation')
subplot(344)
imagesc(reshape(modes*NetZe,[nPxPup nPxPup]));axis image;colorbar
title('Network estimation')
subplot(345)
imagesc(y);axis image;colorbar
title('Network estimation')


subplot(346)
imagesc(x-reshape(modes*Zg,[nPxPup nPxPup]));axis image;colorbar
title('res')
subplot(347)
imagesc(x-reshape(modes*PyrZe,[nPxPup nPxPup]));axis image;colorbar
title('res')
subplot(348)
imagesc(x-reshape(modes*NetZe,[nPxPup nPxPup]));axis image;colorbar
title('res')
subplot(3,4,[9:12])
hold on
plot(Zg,'k')
plot(PyrZe,'r')
plot(NetZe,'b')
legend('Groundtruth','Traditional Pyramid','Pyramid with preconditioner')
grid on; box on


%%

% figure,
% subplot(1,2,1)
% imagesc(zeros(512,512));axis image;colorbar
% subplot(1,2,2)
% imagesc(fftshift(OL1_trained));axis image;colorbar