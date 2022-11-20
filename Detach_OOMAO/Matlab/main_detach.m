addpath functions
clear all

preFold = "../Preconditioners/nocap/base/checkpoint/OL1_R128_M0_RMSE0.029_Epoch_85.mat";

binning       = 1;
D             = 8;
modulation    = 1;
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

jModes = [2:60];

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

[PyrCM,PyrI_0,PyrIM] = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,Samp...
    ,modulation,rooftop,alpha,pupil,OL1_trained,0);
        
%% Meas
r0            = 1.1;
ReadoutNoise = 0;
PhotonNoise = 0;
nPhotonBackground = 0.1;
quantumEfficiency = 1;
atm = GenerateAtmosphereParameters(nLenslet,D,binning,r0,L0,fR0,modulation,fovInPixel,resAO,Samp,nPxPup,pupil);

[x,Zg] = ComputePhaseScreen(atm,PhaseCM);
% [x,Zg] = GenerateFourierPhase(nLenslet,D,binning,r0,L0,fR0,modulation,...
%     fovInPixel,resAO,Samp,nPxPup,pupil,jModes,modes,PhaseCM);

y = PropagatePyr(fovInPixel,x,Samp,modulation,rooftop,alpha,pupil,nPxPup,OL1_trained,1);
if PhotonNoise
y = AddPhotonNoise(y,nPhotonBackground,quantumEfficiency);
end
y = y +randn(size(y)).*ReadoutNoise;
Net_y = y/sum(y(:))-NetI_0;
% Estimation
NetZe = NetCM*Net_y(:);

y = PropagatePyr(fovInPixel,x,Samp,modulation,rooftop,alpha,pupil,nPxPup,OL1_trained,0);
if PhotonNoise
y = AddPhotonNoise(y,nPhotonBackground,quantumEfficiency);
end
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
imagesc(Net_y);axis image;colorbar
title('Network estimation')

res1 = x-reshape(modes*Zg,[nPxPup nPxPup]);
res2 = x-reshape(modes*PyrZe,[nPxPup nPxPup]);
res3 = x-reshape(modes*NetZe,[nPxPup nPxPup]);
resmax = max(max(max(cat(3,res1,res2,res3))));
resmin = min(min(min(cat(3,res1,res2,res3))));
subplot(346)
imagesc(res1,[resmin resmax]);axis image;colorbar
title(sprintf("Phase res $\\sigma = %2.2f$",std(res1(:))),'interpreter','latex','FontSize',15)
subplot(347)
imagesc(res2,[resmin resmax]);axis image;colorbar
title(sprintf("Pyr res $\\sigma = %2.2f$",std(res2(:))),'interpreter','latex','FontSize',15)
subplot(348)
imagesc(res3,[resmin resmax]);axis image;colorbar
title(sprintf("Pyr+DE res $\\sigma = %2.2f$",std(res3(:))),'interpreter','latex','FontSize',15)
subplot(3,4,[9:12])
hold on
plot(Zg,'k','linewidth',2)
plot(PyrZe,'r','linewidth',2)
plot(NetZe,'b','linewidth',2)
legend('Groundtruth','Traditional Pyramid','Pyramid with preconditioner')
grid on; box on


%%

% figure,
% subplot(1,2,1)
% imagesc(zeros(512,512));axis image;colorbar
% subplot(1,2,2)
% imagesc(fftshift(OL1_trained));axis image;colorbar