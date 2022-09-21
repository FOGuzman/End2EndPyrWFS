addpath functions
clear all
binning       = 1;
D             = 1.5;
modulation    = 1;
nLenslet      = 16;
resAO         = 2*nLenslet+1;
r0            = 0.5;
L0            = 20;
fR0           = 1;
noiseVariance = 0.7;
n_lvl         = 0.2;             % noise level in rad^2
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



%% Pyramid calibration
modes = CreateZernikePolynomials(nPxPup,jModes,pupil~=0);
flatMode = CreateZernikePolynomials(nPxPup,1,pupil~=0);


%% Test
% Phase inversion

load('OL1_R128_M0_RMSE0.054.mat');OL1_trained = OL1;

[NetCM,NetI_0] = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,Samp...
    ,modulation,rooftop,alpha,pupil,OL1_trained,1);

[PyrCM,PyrI_0] = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,Samp...
    ,modulation,rooftop,alpha,pupil,OL1_trained,0);
        
%% Meas
jz = 3;
interval = linspace(-2,2,50);


Zp = zeros(1,50);
Zn = Zp;
for k = 1:length(interval)
Zg = zeros(1,35);
Zg(jz) = interval(k);
x = reshape(modes*Zg',[nPxPup nPxPup]);



y = PropagatePyr(fovInPixel,x,Samp,modulation,rooftop,alpha,pupil,nPxPup,OL1_trained,1);
Net_y = y/sum(y(:))-NetI_0;
% Estimation
NetZe = NetCM*Net_y(:);

y = PropagatePyr(fovInPixel,x,Samp,modulation,rooftop,alpha,pupil,nPxPup,OL1_trained,0);
Pyr_y = y/sum(y(:))-PyrI_0;
% Estimation
PyrZe = PyrCM*Pyr_y(:);

Zp(k) = PyrZe(jz);
Zn(k) = NetZe(jz);


end
ZGT = interval;


figure
hold on
plot(ZGT,ZGT,'k','LineWidth',2)
plot(ZGT,Zp,'r','LineWidth',2)
plot(ZGT,Zn,'b','LineWidth',2)
grid on;box on
xlabel('Amplitude')
ylabel('Estimation')
legend('Reference','Traditional Pyramid','Pyramid with preconditioner')
title(['Lineal response Z_' num2str(jz+1)])