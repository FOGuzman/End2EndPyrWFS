addpath tools/functions
clear all;clc;close all

%% Preconditioners paths
%DPWFS_path = "../Preconditioners/nocap/base/checkpoint/OL1_R128_M0_RMSE0.0285_Epoch_92.mat";
DPWFS_path = "/home/fg/Desktop/FOGuzman/End2EndPyrWFS/Detach_OOMAO/Pytorch/training_results/Paper/06-07-2023/r1/DE/DE_Epoch_99_R128_M0_S2_RMSE_0.05316.mat";
%DPWFSn_path = "../Preconditioners/nocap/pnoise/checkpoint/OL1_R128_M0_RMSE0.05275_Epoch_118.mat";
DPWFSn_path = "/home/fg/Desktop/FOGuzman/End2EndPyrWFS/Detach_OOMAO/Pytorch/training_results/Paper/06-07-2023/n1/DE/DE_Epoch_99_R128_M0_S2_RMSE_0.09754.mat";


FigurePath = "./figures/Figure5/";if ~exist(FigurePath, 'dir'), mkdir(FigurePath); end
FigureName1 = "ElementA.pdf";
FigureName2 = "ElementB.pdf";
%% Phisycal parameters
run("./tools/experiments_settings/F4_settings.m")

%% Test parameters
radial_order = 30;
physicalParams.jModes = cumsum(kron([2:2:radial_order*2],[1 1]));
physicalParams.jModes = physicalParams.jModes(2:radial_order+1);
physicalParams.modes = CreateZernikePolynomials(physicalParams.nPxPup,physicalParams.jModes,physicalParams.pupil~=0);

% Test
% Phase inversion
PhaseCM = pinv(physicalParams.modes);
load(DPWFS_path);DPWFS_DE = OL1;
load(DPWFSn_path);DPWFSn_DE = OL1;

pyrMask = fftshift(angle(create_pyMask(physicalParams.fovInPixel,physicalParams.rooftop,physicalParams.alpha)));


Mods = [0:2];


%%
mv = create_pyMask(physicalParams.fovInPixel,physicalParams.rooftop,physicalParams.alpha);
m1 = mv.*exp(1j.*DPWFS_DE);
m2 = mv.*exp(1j.*DPWFSn_DE);
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

x2 = x/norm(x,2);
Dv(m,k) = Linearity(x2,nTheta,fftPhasor,mv,physicalParams);
D1(m,k) = Linearity(x2,nTheta,fftPhasor,m1,physicalParams);
D2(m,k) = Linearity(x2,nTheta,fftPhasor,m2,physicalParams);


SDv(m,k) = SDFactor(Sv(m,k),Dv(m,k),1.3);
SD1(m,k) = SDFactor(S1(m,k),D1(m,k),1.3);
SD2(m,k) = SDFactor(S2(m,k),D2(m,k),1.3);

end

end
Dv = Dv.^-1;
D1 = D1.^-1;
D2 = D2.^-1;
%% Fullplot

lbltxt{1} = sprintf("PWFS, Mod $= %i\\lambda/D_0$   ",Mods(1));
lbltxt{2} = sprintf("PWFS, Mod $= %i\\lambda/D_0$   ",Mods(2));
lbltxt{3} = sprintf("PWFS, Mod $= %i\\lambda/D_0$   ",Mods(3));
lbltxt{4} = sprintf("DPWFS, Mod $= %i\\lambda/D_0$",Mods(1));

ylimit = sort([floor(min(min([Dv(:) D1(:) D2(:)]))) ceil(max(max([Sv(:) S1(:) S2(:)])))+5 ]);
% ylimit = [1e-2 1e2]
ZerLen = length(physicalParams.jModes);

lw = 1;
fig1 = figure('Color','w','Position',[409 169 454 420]);
hold on
plot(1:ZerLen,Sv(1,:),'--r','LineWidth',lw)
plot(1:ZerLen,Sv(2,:),'--g','LineWidth',lw)
plot(1:ZerLen,Sv(3,:),'--b','LineWidth',lw)
plot(1:ZerLen,S1(1,:),'-r','LineWidth',lw)


plot(1:ZerLen,SDv(1,:),'--r','LineWidth',lw)
plot(1:ZerLen,SDv(2,:),'--g','LineWidth',lw)
plot(1:ZerLen,SDv(3,:),'--b','LineWidth',lw)
plot(1:ZerLen,SD1(1,:),'-r','LineWidth',lw)


plot(1:ZerLen,Dv(1,:),'--r','LineWidth',lw)
plot(1:ZerLen,Dv(2,:),'--g','LineWidth',lw)
plot(1:ZerLen,Dv(3,:),'--b','LineWidth',lw)
plot(1:ZerLen,D1(1,:),'-r','LineWidth',lw)

xlabel("Zernike radial order",'interpreter','latex')
set(gca,'XScale','log','YScale','log','FontSize',20,'TickLabelInterpreter','latex','LineWidth',1)
box on
% ylim(ylimit)
leg = legend(lbltxt,'FontSize',10,'interpreter','latex','Position',[0.4841 0.7583 0.4201 0.1655]);



fig2 = figure('Color','w','Position',[865 169 454 420]);
hold on
plot(1:ZerLen,Sv(1,:),'--r','LineWidth',lw)
plot(1:ZerLen,Sv(2,:),'--g','LineWidth',lw)
plot(1:ZerLen,Sv(3,:),'--b','LineWidth',lw)
plot(1:ZerLen,S2(1,:),'-r','LineWidth',lw)

plot(1:ZerLen,SDv(1,:),'--r','LineWidth',lw)
plot(1:ZerLen,SDv(2,:),'--g','LineWidth',lw)
plot(1:ZerLen,SDv(3,:),'--b','LineWidth',lw)
plot(1:ZerLen,SD2(1,:),'-r','LineWidth',lw)

plot(1:ZerLen,Dv(1,:),'--r','LineWidth',lw)
plot(1:ZerLen,Dv(2,:),'--g','LineWidth',lw)
plot(1:ZerLen,Dv(3,:),'--b','LineWidth',lw)
plot(1:ZerLen,D2(1,:),'-r','LineWidth',lw)

xlabel("Zernike radial order",'interpreter','latex')
set(gca,'XScale','log','YScale','log','FontSize',20,'TickLabelInterpreter','latex','LineWidth',1)
box on
%ylim(ylimit)
leg = legend(lbltxt,'FontSize',10,'interpreter','latex','Position',[0.4841 0.7583 0.4201 0.1655]);

exportgraphics(fig1,FigurePath+FigureName1)
exportgraphics(fig2,FigurePath+FigureName2)

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