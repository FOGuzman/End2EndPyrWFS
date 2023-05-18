addpath functions
clear all;clc

%% Preconditioners paths
DPWFS_path = "../Preconditioners/nocap/base/checkpoint/OL1_R128_M0_RMSE0.0285_Epoch_92.mat";
DPWFSn_path = "../Preconditioners/nocap/pnoise/checkpoint/OL1_R128_M0_RMSE0.05275_Epoch_118.mat";
saveFold = "../ComputeResults/Fig8A/";
%% Phisycal parameters
binning       = 1;
D             = 8;
nLenslet      = 16;
resAO         = 2*nLenslet+1;
L0            = 25;
D_R0s         = 40;
R0s           = D./D_R0s;
r0            = R0s;
fR0           = 1;
Samp          = 2;                % OVer-sampling factor
nPxPup        = 128;               % number of pixels to describe the pupil
alpha         = pi/2;
rooftop       = [0,0]; 
fovInPixel    = nPxPup*2*Samp;    % number of pixel to describe the PSD
PyrQ          = zeros(fovInPixel,fovInPixel);
I4Q4          = PyrQ;
nTimes        = fovInPixel/resAO;
N             = 2*Samp*nPxPup;
L             = (N-1)*D/(nPxPup-1);
pupil         = CreatePupil(nPxPup,"disc");
jModes        = 2:60;
ReadoutNoise  = 0;
PhotonNoise   = 1;
quantumEfficiency = 1;

%% Testing parameters
tpr0  = 600;    % test per r0
njumps = 20;
nLims = [0 0.4];
nInterval = linspace(nLims(1),nLims(2),njumps);
Mods = [0];

fprintf("Test range %i\n",njumps);
fprintf("D/r0 = [");
fprintf("%.0f ",D_R0s);
fprintf("]\n");
fprintf("Phothon noise = [");
for k = 1:njumps;fprintf("%.2f ",nInterval(k));end
fprintf("]\n");
fprintf("Modulations = [");
for k = 1:length(Mods);fprintf("%.0f ",Mods(k));end
fprintf("]\n");

%% Vectors and operators
rmse = @(x,y) sqrt(mse(x(:),y(:)));

%% Pyramid calibration
modes = CreateZernikePolynomials(nPxPup,jModes,pupil~=0);
flatMode = CreateZernikePolynomials(nPxPup,1,pupil~=0);

for mc = 1:length(Mods)
modulation    = Mods(mc);
% Matrix fittings
%Phase inversion
PhaseCM = pinv(modes);

load(DPWFS_path);DPWFS_DE = OL1;
load(DPWFSn_path);DPWFSn_DE = OL1;


[PyrCM,PyrI_0,PyrIM] = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,Samp...
    ,modulation,rooftop,alpha,pupil,DPWFS_DE,0);

[DPWFS_CM,DPWFS_I0]  = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,Samp...
    ,modulation,rooftop,alpha,pupil,DPWFS_DE,1);

[DPWFSn_CM,DPWFSn_I0]  = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,Samp...
    ,modulation,rooftop,alpha,pupil,DPWFSn_DE,1);
        
%% start testing
noise_v_meanRMSE_pyr = zeros(1,njumps);
noise_v_meanRMSE_de = zeros(1,njumps);
noise_v_stdRMSE_pyr = zeros(1,njumps);
noise_v_stdRMSE_de = zeros(1,njumps);
for rc = 1:njumps
nPhotonBackground = nInterval(rc);
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

noise_v_meanRMSE_pyr(1,rc) = mean(v_RMSE_pyr);
noise_v_meanRMSE_de(1,rc)  = mean(v_RMSE_de);
noise_v_stdRMSE_pyr(1,rc)  = std(v_RMSE_pyr);
noise_v_stdRMSE_de(1,rc)   = std(v_RMSE_de);


fprintf("Progress:M[%i/%i] = %i | Rnoise[%i/%i] = %.2f | Time per r0 = %.2f seg\n"...
    ,mc,length(Mods),Mods(mc),rc,njumps,nInterval(rc),toc)
end

R(mc).meanRMSEpyr = noise_v_meanRMSE_pyr;
R(mc).stdRMSEpyr = noise_v_stdRMSE_pyr;
R(mc).meanRMSEde = noise_v_meanRMSE_de;
R(mc).stdRMSEde = noise_v_stdRMSE_de;

[x,Zg] = ComputePhaseScreen(atm,PhaseCM);

y = PropagatePyr(fovInPixel,x,Samp,modulation,rooftop,alpha,pupil,nPxPup,OL1_trained,1);
if PhotonNoise
y = AddPhotonNoise(y,nPhotonBackground,quantumEfficiency);
end
y = y +randn(size(y)).*ReadoutNoise;
Net_y = y/sum(y(:));
R(mc).ExampleMeas = Net_y;
end

save(saveFold+"PNoisermseResults_noiser10.mat",'R')



%% Test parameters

%preFold = "../Preconditioners/nocap/base/checkpoint/OL1_R128_M0_RMSE0.0285_Epoch_92.mat";
preFold = "../Preconditioners/nocap/pnoise/checkpoint/OL1_R128_M0_RMSE0.05275_Epoch_118.mat";
saveFold = "../Preconditioners/nocap/pnoise/";

fprintf("Test range %i\n",njumps);
fprintf("D/r0 = [");
fprintf("%.0f ",D_R0s);
fprintf("]\n");
fprintf("Phothon noise = [");
for k = 1:njumps;fprintf("%.2f ",nInterval(k));end
fprintf("]\n");
fprintf("Modulations = [");
for k = 1:length(Mods);fprintf("%.0f ",Mods(k));end
fprintf("]\n");


%% Pyramid calibration
modes = CreateZernikePolynomials(nPxPup,jModes,pupil~=0);
flatMode = CreateZernikePolynomials(nPxPup,1,pupil~=0);

for mc = 1:length(Mods)
modulation    = Mods(mc);

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
noise_v_meanRMSE_pyr = zeros(1,njumps);
noise_v_meanRMSE_de = zeros(1,njumps);
noise_v_stdRMSE_pyr = zeros(1,njumps);
noise_v_stdRMSE_de = zeros(1,njumps);
for rc = 1:njumps

ReadoutNoise = 0;
PhotonNoise = 1;
nPhotonBackground = nInterval(rc);
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

noise_v_meanRMSE_pyr(1,rc) = mean(v_RMSE_pyr);
noise_v_meanRMSE_de(1,rc)  = mean(v_RMSE_de);
noise_v_stdRMSE_pyr(1,rc)  = std(v_RMSE_pyr);
noise_v_stdRMSE_de(1,rc)   = std(v_RMSE_de);


fprintf("Progress:M[%i/%i] = %i | Rnoise[%i/%i] = %.2f | Time per r0 = %.2f seg\n"...
    ,mc,length(Mods),Mods(mc),rc,njumps,nInterval(rc),toc)
end

R(mc).meanRMSEpyr = noise_v_meanRMSE_pyr;
R(mc).stdRMSEpyr = noise_v_stdRMSE_pyr;
R(mc).meanRMSEde = noise_v_meanRMSE_de;
R(mc).stdRMSEde = noise_v_stdRMSE_de;

[x,Zg] = ComputePhaseScreen(atm,PhaseCM);

y = PropagatePyr(fovInPixel,x,Samp,modulation,rooftop,alpha,pupil,nPxPup,OL1_trained,1);
if PhotonNoise
y = AddPhotonNoise(y,nPhotonBackground,quantumEfficiency);
end
y = y +randn(size(y)).*ReadoutNoise;
Net_y = y/sum(y(:));
R(mc).ExampleMeas = Net_y;
end

save(saveFold+"PNoisermseResults_noiser10.mat",'R')





%% Plot
loadFold = "../Preconditioners/nocap/base/";
R1 =load(loadFold+"PNoisermseResults_noiser10.mat");R1=R1.R;
loadFold = "../Preconditioners/nocap/pnoise/";
R2 =load(loadFold+"PNoisermseResults_noiser10.mat");R2=R2.R;
mod = 1;
lbltxt{1} = sprintf("PWFS");
lbltxt{2} = sprintf("DPWFS");
lbltxt{3} = sprintf("DPWFS*");
fig = figure('Color','w');
errorbar(nInterval,R1(mod).meanRMSEpyr,R1(mod).stdRMSEpyr,'r','LineWidth',1.5)
hold on
errorbar(nInterval,R1(mod).meanRMSEde,R1(mod).stdRMSEde,'g','LineWidth',1.5)
errorbar(nInterval,R2(mod).meanRMSEde,R2(mod).stdRMSEde,'b','LineWidth',1.5)
xlabel("Photon noise",'Interpreter','latex')
ylabel("RMSE [radians]",'Interpreter','latex')
set(gca,'FontSize',13,'TickLabelInterpreter','latex','LineWidth',1)
leg = legend(lbltxt,'interpreter','latex','Location','northwest');
ylim([0 0.1])

fold = "./figures/v5/";
name = "Pnoise_PerformanceV2.pdf";
exportgraphics(fig,fold+name)



%% Functions

function [outputArg1,outputArg2] = Prop(inputArg1,inputArg2)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
outputArg1 = inputArg1;
outputArg2 = inputArg2;
end