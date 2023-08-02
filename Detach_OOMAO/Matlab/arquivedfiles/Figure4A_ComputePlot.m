addpath tools/functions
clear all;clc;close all

%% Preconditioners paths
%DPWFS_path = "../Preconditioners/nocap/base/checkpoint/OL1_R128_M0_RMSE0.0285_Epoch_92.mat";
DPWFS_path = "/home/fg/Desktop/FOGuzman/End2EndPyrWFS/Detach_OOMAO/Pytorch/training_results/Paper/06-07-2023/pupil/DE/DE_project.mat";
savePath = "./ComputeResults/paper/Fig4A/";if ~exist(savePath, 'dir'), mkdir(savePath); end
matName = "r0PerformanceFig4A";
FigurePath = "./figures/paper/Figure4/";if ~exist(FigurePath, 'dir'), mkdir(FigurePath); end
FigureName = "ElementA.pdf";
Compute = false;


%% Phisycal parameters
run("./tools/experiments_settings/F4_settings.m")

%% Test parameters
if Compute
tpr0  = 500;    % test per r0
physicalParams.D_R0s = [90 80 70 60 50 40 30 20 10 1];%[30 25 20 15 10 8 5 3 1]
physicalParams.R0s = physicalParams.D./physicalParams.D_R0s;
rjumps = length(physicalParams.R0s);
Mods = [0 1 2];
RandNumberSeed = 666;


fprintf("Test range %i\n",rjumps);
fprintf("D/r0 = [");
for k = 1:rjumps;fprintf("%.0f ",physicalParams.D_R0s(k));end
fprintf("]\n");
fprintf("r0 = [");
for k = 1:rjumps;fprintf("%.2f ",physicalParams.R0s(k));end
fprintf("]\n");
fprintf("Modulations = [");
for k = 1:length(Mods);fprintf("%.0f ",Mods(k));end
fprintf("]\n");

% Vectors and operators
rmse = @(x,y) sqrt(mse(x(:),y(:)));




for mc = 1:length(Mods)
physicalParams.modulation    = Mods(mc);

% Matrix fittings
%Phase inversion
PhaseCM = pinv(physicalParams.modes);

load(DPWFS_path);DPWFS_DE = OL1;

[PyrCM,PyrI_0,PyrIM]   = PyrCalibration(physicalParams,DPWFS_DE,0);
[DPWFS_CM,DPWFS_I0]    = PyrCalibration(physicalParams,DPWFS_DE,1);
        
%% start testing

%Result vector mean(1,:), std(2,:). Noise level (:,1,2,...,N)
RMSEpyr    = zeros(2,rjumps);
RMSEdpwfs  = zeros(2,rjumps);

rng(RandNumberSeed)
for rc = 1:rjumps
physicalParams.r0            = physicalParams.R0s(rc);
atm = GenerateAtmosphereParameters(physicalParams);

%
v_RMSE_pyr     = zeros(1,tpr0);
v_RMSE_dpwfs  = zeros(1,tpr0);

tic
for tc = 1:tpr0
[xin,Zg] = ComputePhaseScreen(atm,PhaseCM);

[Zpyr]    = PropAndSamp(physicalParams,xin,DPWFS_DE ,PyrI_0   ,PyrCM    ,0);
[Zdpwfs]  = PropAndSamp(physicalParams,xin,DPWFS_DE ,DPWFS_I0 ,DPWFS_CM ,1);

v_RMSE_pyr(1,tc)     = rmse(Zg,Zpyr);
v_RMSE_dpwfs(1,tc)   = rmse(Zg,Zdpwfs);

end

RMSEpyr(1,rc) = mean(v_RMSE_pyr);
RMSEpyr(2,rc) = std(v_RMSE_pyr);

RMSEdpwfs(1,rc) = mean(v_RMSE_dpwfs);
RMSEdpwfs(2,rc) = std(v_RMSE_dpwfs);

fprintf("Progress:M[%i/%i] = %i | r0[%i/%i] = %.2f | Time per r0 = %.2f seg\n"...
    ,mc,length(Mods),Mods(mc),rc,rjumps,physicalParams.R0s(rc),toc)
end

Results{mc}.RMSEpyr    = RMSEpyr;
Results{mc}.RMSEdpwfs  = RMSEdpwfs;

end

%% Create log
INFO = physicalParams;
INFO.r0s  = physicalParams.D_R0s;
INFO.date               = date;
INFO.datapointsPerLevel = tpr0;
INFO.RandNumberSeed = RandNumberSeed;
INFO.FilesAndPathds = {DPWFS_path,savePath,matName,FigurePath,FigureName}';
for k = 1:length(Mods);Results{k}.INFO = INFO;Results{k}.INFO.modulation = Mods(k);end

save(savePath+matName+".mat",'Results')
end
%% Plot

if ~exist(savePath+matName+".mat", 'file')
    warning('There is no computed files on the folder: Change "Compute" to false and recalculate');
    return;  % Finish the script
end

Rin = load(savePath+matName+".mat");R=Rin.Results;

r0s = R{1}.INFO.D_R0s;
y1 = R{1}.RMSEpyr(1,:);
y2 = R{2}.RMSEpyr(1,:);
y3 = R{3}.RMSEpyr(1,:);
y4 = R{4}.RMSEpyr(1,:);
y5 = R{1}.RMSEdpwfs(1,:);

lbltxt{1} = sprintf("PWFS-M%i",R{1}.INFO.modulation);
lbltxt{2} = sprintf("PWFS-M%i",R{2}.INFO.modulation);
lbltxt{3} = sprintf("PWFS-M%i",R{3}.INFO.modulation);
lbltxt{4} = sprintf("PWFS-M%i",R{4}.INFO.modulation);
lbltxt{5} = sprintf("DPWFS-R1");

fig = figure('Color','w','Units','normalized','Position',[0.5436 0.1528 0.4427 0.6331]);

plot(r0s,y1,'--dr','LineWidth',1.5,'MarkerFaceColor','r')
hold on
plot(r0s,y2,'--dg','LineWidth',1.5,'MarkerFaceColor','g')
plot(r0s,y3,'--db','LineWidth',1.5,'MarkerFaceColor','b')
plot(r0s,y4,'--dm','LineWidth',1.5,'MarkerFaceColor','m')
plot(r0s,y5,'-or','LineWidth',1.5,'MarkerFaceColor','r')
set(gca,'XDir','reverse','FontSize',28,'TickLabelInterpreter','latex')
xlabel('$D/r_0$','interpreter','latex','FontSize',22)
ylabel('RMSE','interpreter','latex','FontSize',22)
legend(lbltxt,'interpreter','latex','FontSize',19)
xlim([min(r0s) max(r0s)])





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