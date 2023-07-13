addpath tools/functions
clear all;clc;close all

%% Preconditioners paths
DPWFSr1_path = "../Pytorch/training_results/Paper/06-07-2023/original.mat";
DPWFSr2_path = "../Pytorch/training_results/Paper/06-07-2023/r2_nico.mat";
DPWFSn1_path = "../Pytorch/training_results/Paper/06-07-2023/n1_nico.mat";
savePath = "./ComputeResults/paper/Fig5/";if ~exist(savePath, 'dir'), mkdir(savePath); end
matName = "r0PerformanceFig5A";
FigurePath = "./figures/paper/Figure5/";if ~exist(FigurePath, 'dir'), mkdir(FigurePath); end
FigureNameA = "ElementA.pdf";
FigureNameB = "ElementB.pdf";
Compute = false;


%% Phisycal parameters
run("./tools/experiments_settings/F5_settings.m")

%% Test parameters
if Compute
tpr0  = 500;    % test per r0
physicalParams.D_R0s = [90 80 70 60 50 40 30 20 10 1];%[30 25 20 15 10 8 5 3 1]
physicalParams.R0s = physicalParams.D./physicalParams.D_R0s;
rjumps = length(physicalParams.R0s);
Mods = [0 3];
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

load(DPWFSr1_path);DPWFS_DE = OL1;

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
INFO.FilesAndPathds = {DPWFSr1_path,savePath,matName,FigurePath,FigureNameA}';
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
y5 = R{1}.RMSEdpwfs(1,:);
y6 = R{1}.RMSEdpwfs2(1,:);
y7 = R{1}.RMSEdpwfs3(1,:);

lbltxt{1} = sprintf("PWFS-M%i",R{1}.INFO.modulation);
lbltxt{2} = sprintf("PWFS-M%i",R{2}.INFO.modulation);
lbltxt{3} = sprintf("DPWFS-R1");
lbltxt{4} = sprintf("DPWFS-R2");
lbltxt{5} = sprintf("DPWFS-N1");

fig = figure('Color','w','Units','normalized','Position',[0.5436 0.1528 0.4427 0.6331]);

plot(r0s,y1,'--dr','LineWidth',1.5,'MarkerFaceColor','r')
hold on
plot(r0s,y2,'--dm','LineWidth',1.5,'MarkerFaceColor','m')
plot(r0s,y5,'-r','LineWidth',1.5,'MarkerFaceColor','r')
plot(r0s,y6,'-g','LineWidth',1.5,'MarkerFaceColor','b')
plot(r0s,y7,'-b','LineWidth',1.5,'MarkerFaceColor','m')
set(gca,'XDir','reverse','FontSize',20)
xlabel('$D/r_0$','interpreter','latex','FontSize',22)
ylabel('RMSE','FontSize',22)
legend(lbltxt,'FontSize',19)
xlim([min(r0s) max(r0s)])

exportgraphics(fig,FigurePath+FigureNameA)


%% SD
radial_order = 30;
physicalParams.jModes = cumsum(kron([2:2:radial_order*2],[1 1]));
physicalParams.jModes = physicalParams.jModes(2:radial_order+1);
physicalParams.modes = CreateZernikePolynomials(physicalParams.nPxPup,physicalParams.jModes,physicalParams.pupil~=0);

% Test
% Phase inversion
PhaseCM = pinv(physicalParams.modes);
load(DPWFSr1_path);DPWFS_R1 = OL1;
load(DPWFSr2_path);DPWFS_R2 = OL1;
load(DPWFSn1_path);DPWFS_N1 = OL1;

pyrMask = fftshift(angle(create_pyMask(physicalParams.fovInPixel,physicalParams.rooftop,physicalParams.alpha)));


Mods = [0:3];


%%
mv = create_pyMask(physicalParams.fovInPixel,physicalParams.rooftop,physicalParams.alpha);
m1 = mv.*exp(1j.*DPWFS_R1);
m2 = mv.*exp(1j.*DPWFS_R2);
m3 = mv.*exp(1j.*DPWFS_N1);
for m = 1:length(Mods)
physicalParams.modulation = Mods(m);
nTheta = round(2*pi*physicalParams.Samp*physicalParams.modulation);
fftPhasor = GetModPhasor(physicalParams.nPxPup,physicalParams.Samp,physicalParams.modulation);
for k = 1:length(physicalParams.jModes)
x = reshape(physicalParams.modes(:,k),[physicalParams.nPxPup physicalParams.nPxPup]);
x1 = x/norm(x,2);
Sv(m,k) = Sensitivity(x1,nTheta,fftPhasor,mv,physicalParams);
S1(m,k) = Sensitivity(x1,nTheta,fftPhasor,m1,physicalParams);
S2(m,k) = Sensitivity(x1,nTheta,fftPhasor,m2,physicalParams);
S3(m,k) = Sensitivity(x1,nTheta,fftPhasor,m3,physicalParams);

x2 = x/norm(x,2);
Dv(m,k) = Linearity(x2,nTheta,fftPhasor,mv,physicalParams);
D1(m,k) = Linearity(x2,nTheta,fftPhasor,m1,physicalParams);
D2(m,k) = Linearity(x2,nTheta,fftPhasor,m2,physicalParams);
D3(m,k) = Linearity(x2,nTheta,fftPhasor,m3,physicalParams);

SDv(m,k) = SDFactor(Sv(m,k),Dv(m,k),1.3);
SD1(m,k) = SDFactor(S1(m,k),D1(m,k),1.3);
SD2(m,k) = SDFactor(S2(m,k),D2(m,k),1.3);
SD3(m,k) = SDFactor(S3(m,k),D3(m,k),1.3);

end

end
Dv = Dv.^-1;
D1 = D1.^-1;
D2 = D2.^-1;
D3 = D3.^-1;
%% Fullplot

lbltxt{1} = sprintf("PWFS-M%i",R{1}.INFO.modulation);
lbltxt{2} = sprintf("PWFS-M%i",R{2}.INFO.modulation);
lbltxt{3} = sprintf("DPWFS-R1");
lbltxt{4} = sprintf("DPWFS-R2");
lbltxt{5} = sprintf("DPWFS-N1");

ylimit = sort([floor(min(min([Dv(:) D1(:)]))) ceil(max(max([Sv(:) S1(:)])))+5 ]);
% ylimit = [1e-2 1e2]
ZerLen = length(physicalParams.jModes);

lw = 1;
fig1 = figure('Color','w','Units','normalized','Position',[0.3141 0.2120 0.3766 0.6102]);
hold on
plot(1:ZerLen,Sv(1,:),'--r','LineWidth',lw)
plot(1:ZerLen,Sv(2,:),'--m','LineWidth',lw)
plot(1:ZerLen,S1(1,:),'-r','LineWidth',lw)
plot(1:ZerLen,S2(1,:),'-g','LineWidth',lw)
plot(1:ZerLen,S3(1,:),'-b','LineWidth',lw)


plot(1:ZerLen,SDv(1,:),'--r','LineWidth',lw)
plot(1:ZerLen,SDv(2,:),'--m','LineWidth',lw)
plot(1:ZerLen,SD1(1,:),'-r','LineWidth',lw)
plot(1:ZerLen,SD2(1,:),'-g','LineWidth',lw)
plot(1:ZerLen,SD3(1,:),'-b','LineWidth',lw)

plot(1:ZerLen,Dv(1,:),'--r','LineWidth',lw)
plot(1:ZerLen,Dv(2,:),'--m','LineWidth',lw)
plot(1:ZerLen,D1(1,:),'-r','LineWidth',lw)
plot(1:ZerLen,D2(1,:),'-g','LineWidth',lw)
plot(1:ZerLen,D3(1,:),'-b','LineWidth',lw)

xlabel("Zernike radial order",'interpreter','latex')
set(gca,'XScale','log','YScale','log','FontSize',20,'LineWidth',1)
box on; grid on
% ylim(ylimit)
leg = legend(lbltxt,'FontSize',12,'Position',[0.6731 0.7165 0.2334 0.2086]);

exportgraphics(fig1,FigurePath+FigureNameB)

%% FUNCTIONS

function s = Sensitivity(x,nTheta,fftPhasor,pyrMask,params)
Samp = params.Samp;
nPxPup = params.nPxPup;
pupil = params.pupil;

fovInPixel = nPxPup*2*Samp;
Fm = fftshift(fft2(pyrMask));
Ws = ones(1,nTheta)/nTheta;
Pup = zeros(fovInPixel);
[n1,n2,n3] = size(pupil);
u = fix(2+nPxPup*(2*Samp-1)/2:nPxPup*(2*Samp+1)/2+1);
Pup(u,u,:) = reshape(pupil, n1,[],1);

phi_i = zeros(fovInPixel);
phi_i(u,u,:) = reshape(x, n1,[],1);
if nTheta == 0;nTheta = 1;end

Imq = zeros(fovInPixel);
for k = 1:nTheta
%f1 = conv2(Pup.*fftPhasor(:,:,k),Fm,'same');
f1 = ifft2(fft2(Pup.*fftPhasor(:,:,k)).*fft2(Fm));
%f2 =  conv2(Pup.*fftPhasor(:,:,k).*phi_i,Fm,'same');
f2 =  ifft2(fft2(Pup.*fftPhasor(:,:,k).*phi_i).*fft2(Fm));
Imq = Imq + (2*imag(f1.*conj(f2)))/nTheta;
end

s = norm(Imq,2);%./norm(phi_i,2);

end



function d = Linearity(x,nTheta,fftPhasor,pyrMask,params)
Samp = params.Samp;
nPxPup = params.nPxPup;
pupil = params.pupil;

fovInPixel = nPxPup*2*Samp;
Fm = fftshift(fft2(pyrMask));
Ws = ones(1,nTheta)/nTheta;
Pup = zeros(fovInPixel);
[n1,n2,n3] = size(pupil);
u = fix(2+nPxPup*(2*Samp-1)/2:nPxPup*(2*Samp+1)/2+1);
Pup(u,u,:) = reshape(pupil, n1,[],1);

phi_i = zeros(fovInPixel);
phi_i(u,u,:) = reshape(x, n1,[],1);
if nTheta == 0;nTheta = 1;end

% A part
ImqA = zeros(fovInPixel);
for k = 1:nTheta
%f1 = conv2(Pup.*fftPhasor(:,:,k),Fm,'same');
f1 = real(abs(ifft2(fft2(Pup.*fftPhasor(:,:,k).*phi_i).*fft2(Fm))).^2);

ImqA = ImqA + f1/nTheta;
end

ImqB = zeros(fovInPixel);
for k = 1:nTheta
%f1 = conv2(Pup.*fftPhasor(:,:,k),Fm,'same');
f1 = ifft2(fft2(Pup.*fftPhasor(:,:,k)).*fft2(Fm));
%f2 =  conv2(Pup.*fftPhasor(:,:,k).*phi_i,Fm,'same');
f2 =  ifft2(fft2(Pup.*fftPhasor(:,:,k).*phi_i.^2).*fft2(Fm));

ImqB = ImqB + (real(f1.*conj(f2)))/nTheta;
end

d = norm(ImqA-ImqB,2);

end


function [sd] = SDFactor(s,d,nu)
sd = s.^nu.*d.^(-1/nu);
end


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