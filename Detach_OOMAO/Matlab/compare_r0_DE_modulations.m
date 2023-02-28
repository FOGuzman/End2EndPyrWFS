addpath functions
clear all;clc

preFold = "../Preconditioners/nocap/mod/OL1_R64_M2_RMSE0.005704_Epoch_109.mat";
preFold = "../Preconditioners/nocap/base/checkpoint/OL1_R128_M0_RMSE0.0285_Epoch_92.mat";
%preFold = "../Preconditioners/nocap/pnoise/checkpoint/OL1_R128_M0_RMSE0.0538_Epoch_106.mat";


binning       = 1;
D             = 8;

nLenslet      = 16;
resAO         = 2*nLenslet+1;
L0            = 25;
fR0           = 1;
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
jModes        = [2:60];

%% Test parameters
saveFold = "../Preconditioners/nocap/base/";
tpr0  = 1000;    % test per r0
D_R0s = [30 25 20 15 10 8 5 3 1];%[30 25 20 15 10 8 5 3 1]
R0s = D./D_R0s;
rjumps = length(R0s);
Mods = [0 1 2];

fprintf("Test range %i\n",rjumps);
fprintf("D/r0 = [");
for k = 1:rjumps;fprintf("%.0f ",D_R0s(k));end
fprintf("]\n");
fprintf("r0 = [");
for k = 1:rjumps;fprintf("%.2f ",R0s(k));end
fprintf("]\n");
fprintf("Modulations = [");
for k = 1:length(Mods);fprintf("%.0f ",Mods(k));end
fprintf("]\n");

%% Vectors and operators
rmse = @(x,y) sqrt(mse(x(:),y(:)));
R.DE = preFold;
R.R0s = R0s;
R.D_R0s = D_R0s;
for mc = 1:length(Mods)
R(mc).modulation = Mods(mc);
end






for mc = 1:length(Mods)
modulation    = Mods(mc);



%% Pyramid calibration
modes = CreateZernikePolynomials(nPxPup,jModes,pupil~=0);
flatMode = CreateZernikePolynomials(nPxPup,1,pupil~=0);


%% Matrix fittings
% Phase inversion
IM = [];     
for k = 1:length(jModes)
imMode = reshape(modes(:,k),[nPxPup nPxPup]);
IM = [IM imMode(:)];
end
PhaseCM = pinv(IM);

load(preFold);OL1_trained = OL1;

[NetCM,NetI_0] = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,Samp...
    ,modulation,rooftop,alpha,pupil,OL1_trained,1);

[PyrCM,PyrI_0,PyrIM] = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,Samp...
    ,modulation,rooftop,alpha,pupil,OL1_trained,0);
        
%% start testing
r0_v_meanRMSE_pyr = zeros(1,rjumps);
r0_v_meanRMSE_de = zeros(1,rjumps);
r0_v_stdRMSE_pyr = zeros(1,rjumps);
r0_v_stdRMSE_de = zeros(1,rjumps);
for rc = 1:rjumps
r0            = R0s(rc);
ReadoutNoise = 1;
PhotonNoise = 1;
nPhotonBackground = 0.3;
quantumEfficiency = 1;
atm = GenerateAtmosphereParameters(nLenslet,D,binning,r0,L0,fR0,modulation,fovInPixel,resAO,Samp,nPxPup,pupil);

%
v_RMSE_pyr = zeros(1,tpr0);
v_RMSE_de = zeros(1,tpr0);
tic
parfor tc = 1:tpr0
[x,Zg] = ComputePhaseScreen(atm,PhaseCM);

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

v_RMSE_pyr(1,tc) = rmse(Zg,PyrZe);
v_RMSE_de(1,tc)  = rmse(Zg,NetZe);

end

r0_v_meanRMSE_pyr(1,rc) = mean(v_RMSE_pyr);
r0_v_meanRMSE_de(1,rc)  = mean(v_RMSE_de);
r0_v_stdRMSE_pyr(1,rc)  = std(v_RMSE_pyr);
r0_v_stdRMSE_de(1,rc)   = std(v_RMSE_de);


fprintf("Progress:M[%i/%i] = %i | r0[%i/%i] = %.2f | Time per r0 = %.2f seg\n"...
    ,mc,length(Mods),Mods(mc),rc,rjumps,R0s(rc),toc)
end

R(mc).meanRMSEpyr = r0_v_meanRMSE_pyr;
R(mc).stdRMSEpyr = r0_v_stdRMSE_pyr;
R(mc).meanRMSEde = r0_v_meanRMSE_de;
R(mc).stdRMSEde = r0_v_stdRMSE_de;
R(mc).ReadoutNoise = ReadoutNoise;
R(mc).PhotonNoise = PhotonNoise;
R(mc).nPhotonBackground = nPhotonBackground;
R(mc).quantumEfficiency = quantumEfficiency;


[x,Zg] = ComputePhaseScreen(atm,PhaseCM);

y = PropagatePyr(fovInPixel,x,Samp,modulation,rooftop,alpha,pupil,nPxPup,OL1_trained,1);
if PhotonNoise
y = AddPhotonNoise(y,nPhotonBackground,quantumEfficiency);
end
y = y +randn(size(y)).*ReadoutNoise;
Net_y = y/sum(y(:));
R(mc).ExampleMeas = Net_y;
end

save(saveFold+"rmseResults_R0_noise_zoom2.mat",'R')

%% Plot
compare_r0_DE_modulations_plot();

fold = "./figures/v3/";
name = "R0_result2N_zoom_v2.pdf";
exportgraphics(fig,fold+name)