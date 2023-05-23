addpath tools/functions
clear all;clc;close all

%% Preconditioners paths (Compute true to recalculate)
DPWFS_path = "../Preconditioners/nocap/base/checkpoint/OL1_R128_M0_RMSE0.0285_Epoch_92.mat";
%DPWFS_path = "../Preconditioners/nocap/l1/checkpoint/OL1_R128_M0_RMSE0.0285_Epoch_106.mat";
DPWFSn_path = "../Preconditioners/nocap/pnoise/checkpoint/OL1_R128_M0_RMSE0.05275_Epoch_118.mat";
savePath = "./ComputeResults/Fig8A/";if ~exist(savePath, 'dir'), mkdir(savePath); end
matName = "PhothonNoiseFigA";
FigurePath = "./figures/Figure8/";if ~exist(FigurePath, 'dir'), mkdir(FigurePath); end
FigureName = "ElementA.pdf";
Compute = true;
%% Phisycal parameters
run("./tools/experiments_settings/F8A_settings.m")

if Compute
%% Testing parameters
tpr0  = 10000;    % test per r0
njumps = 20;
nLims = [0 0.4];
nInterval = linspace(nLims(1),nLims(2),njumps);
Mods = [0];
RandNumberSeed = 666;

% Print status
fprintf("Test range %i\n",njumps);
fprintf("D/r0 = [");
fprintf("%.0f ",physicalParams.D_R0s);
fprintf("]\n");
fprintf("Phothon noise = [");
for k = 1:njumps;fprintf("%.2f ",nInterval(k));end
fprintf("]\n");
fprintf("Modulations = [");
for k = 1:length(Mods);fprintf("%.0f ",Mods(k));end
fprintf("]\n");

% Vectors and operators
rmse = @(x,y) sqrt(mse(x(:),y(:)));

%% Pyramid calibration

for mc = 1:length(Mods)
physicalParams.modulation    = Mods(mc);
% Matrix fittings
%Phase inversion
PhaseCM = pinv(physicalParams.modes);

load(DPWFS_path);DPWFS_DE = OL1;
load(DPWFSn_path);DPWFSn_DE = OL1;


[PyrCM,PyrI_0,PyrIM]   = PyrCalibration(physicalParams,DPWFS_DE,0);

[DPWFS_CM,DPWFS_I0]    = PyrCalibration(physicalParams,DPWFS_DE,1);

[DPWFSn_CM,DPWFSn_I0]  = PyrCalibration(physicalParams,DPWFSn_DE,1);
        
%% start testing

%Result vector mean(1,:), std(2,:). Noise level (:,1,2,...,N)
RMSEpyr    = zeros(2,njumps);
RMSEdpwfs  = zeros(2,njumps);
RMSEdpwfsn = zeros(2,njumps);
atm = GenerateAtmosphereParameters(physicalParams);
for rc = 1:njumps
physicalParams.nPhotonBackground = nInterval(rc);


%
v_RMSE_pyr     = zeros(1,tpr0);
v_RMSE_dpwfs  = zeros(1,tpr0);
v_RMSE_dpwfsn  = zeros(1,tpr0);

tic
parfor tc = 1:tpr0
[x,Zg] = ComputePhaseScreen(atm,PhaseCM);

[Zpyr]    = PropAndSamp(physicalParams,x,DPWFS_DE ,PyrI_0   ,PyrCM    ,0);
[Zdpwfs]  = PropAndSamp(physicalParams,x,DPWFS_DE ,DPWFS_I0 ,DPWFS_CM ,1);
[Zdpwfsn] = PropAndSamp(physicalParams,x,DPWFSn_DE,DPWFSn_I0,DPWFSn_CM,1);

v_RMSE_pyr(1,tc)     = rmse(Zg,Zpyr);
v_RMSE_dpwfs(1,tc)   = rmse(Zg,Zdpwfs);
v_RMSE_dpwfsn(1,tc)  = rmse(Zg,Zdpwfsn);

end

RMSEpyr(1,rc) = mean(v_RMSE_pyr);
RMSEpyr(2,rc) = std(v_RMSE_pyr);

RMSEdpwfs(1,rc) = mean(v_RMSE_dpwfs);
RMSEdpwfs(2,rc) = std(v_RMSE_dpwfs);

RMSEdpwfsn(1,rc) = mean(v_RMSE_dpwfsn);
RMSEdpwfsn(2,rc) = std(v_RMSE_dpwfsn);


fprintf("Progress: Mod[%i/%i] = %i | Photon Noise[%i/%i] = %.2f | Time per interval = %.2f seg\n"...
    ,mc,length(Mods),Mods(mc),rc,njumps,nInterval(rc),toc)
end

Results(mc).RMSEpyr    = RMSEpyr;
Results(mc).RMSEdpwfs  = RMSEdpwfs;
Results(mc).RMSEdpwfsn = RMSEdpwfsn;

end

%% Create log
INFO = physicalParams;
INFO.nPhotonBackground  = nInterval;
INFO.date               = date;
INFO.datapointsPerLevel = tpr0;
INFO.RandNumberSeed = RandNumberSeed;
INFO.FilesAndPathds = {DPWFS_path,DPWFSn_path,savePath,matName,FigurePath,FigureName}';
Results.INFO = INFO;

save(savePath+matName+".mat",'Results')
end

%% Plot

Rin = load(savePath+matName+".mat");R=Rin.Results;
nlvl = Rin.Results.INFO.nPhotonBackground;
modi = 1;
lbltxt{1} = sprintf("PWFS");
lbltxt{2} = sprintf("DPWFS");
lbltxt{3} = sprintf("DPWFS*");
fig = figure('Color','w');
errorbar(nlvl,R(modi).RMSEpyr(1,:)   ,R(modi).RMSEpyr(2,:)   ,'r','LineWidth',1.5);hold on
errorbar(nlvl,R(modi).RMSEdpwfs(1,:) ,R(modi).RMSEdpwfs(2,:) ,'g','LineWidth',1.5)
errorbar(nlvl,R(modi).RMSEdpwfsn(1,:),R(modi).RMSEdpwfsn(2,:),'b','LineWidth',1.5)
xlabel("Photon noise",'Interpreter','latex')
ylabel("RMSE [radians]",'Interpreter','latex')
set(gca,'FontSize',13,'TickLabelInterpreter','latex','LineWidth',1)
leg = legend(lbltxt,'interpreter','latex','Location','northwest');

ylim([0 0.6])


exportgraphics(fig,FigurePath+FigureName)



%% Functions

function [Ze,y_noise] = PropAndSamp(params,x,DE,I0,CM,flag)

y = PropagatePyr(params,x,DE,flag);
if params.PhotonNoise
y = AddPhotonNoise(y,params.nPhotonBackground,params.quantumEfficiency);
end
y = y +randn(size(y)).*params.ReadoutNoise;
y_noise = y/sum(y(:))-I0;
% Estimation
Ze = CM*y_noise(:);

end