addpath functions
clear all;clc;close all

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


radial_order = 30;
jModes = cumsum(kron([2:2:radial_order*2],[1 1]));
jModes = jModes(2:radial_order+1);
% jModes = [2:100];
% jModes = cumsum(kron([2:2:radial_order*2],[1 1]));
% jModes = jModes(1:radial_order*2);

% jModes1 = 1;
% nx = [1:2:radial_order*2];
% for k = 1:radial_order
%     jModes1 = [jModes1 jModes1(end)+2*nx(k)+1];
% end
% jModes2 = jModes1+[1:2:length(jModes1)*2];
% jModes = sort([jModes1 jModes2]);
% jModes = jModes(2:radial_order+1);




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


Mods = [0:2];

% for m = 1:length(Mods)
%     modulation = Mods(m);
% 
% [NetCM_base,NetI_0_base] = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,Samp...
%     ,modulation,rooftop,alpha,pupil,OL1_base,1);
% 
% [NetCM_noise,NetI_0_noise] = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,Samp...
%     ,modulation,rooftop,alpha,pupil,OL1_noise,1);
% 
% [PyrCM,PyrI_0,PyrIM] = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,Samp...
%     ,modulation,rooftop,alpha,pupil,OL1_base,0);
%         
% %% Zernikes
% 
% for k = 1:length(jModes)
% x = reshape(modes(:,k),[nPxPup nPxPup]);
% 
% y1 = PropagatePyr(fovInPixel,x,Samp,modulation,rooftop,alpha,pupil,nPxPup,OL1_base,1);
% y1 = y1 - NetI_0_base;
% S1(m,k) = norm(y1,2);%/norm(x,2);
% 
% y2 = PropagatePyr(fovInPixel,x,Samp,modulation,rooftop,alpha,pupil,nPxPup,OL1_noise,1);
% y2 = y2 - NetI_0_noise;
% S2(m,k) = norm(y2,2);%/norm(x,2);
% 
% yv = PropagatePyr(fovInPixel,x,Samp,modulation,rooftop,alpha,pupil,nPxPup,OL1_base,0);
% yv = yv - PyrI_0;
% Sv(m,k) = norm(yv,2);%/norm(x,2);
% end
% 
% end


%% Figure

% fig = figure('Color','w');
% hold on
% plot(1:length(jModes),S1(1,:),'-xr')
% plot(1:length(jModes),S1(2,:),'-xg')
% plot(1:length(jModes),S1(3,:),'-xb')
% plot(1:length(jModes),Sv(1,:),'--xr')
% plot(1:length(jModes),Sv(2,:),'--xg')
% plot(1:length(jModes),Sv(3,:),'--xb')


%%


% nTheta = round(2*pi*Samp*modulation);
% fftPhasor = GetModPhasor(nPxPup,Samp,modulation);
% pyrMask = create_pyMask(fovInPixel,rooftop,alpha);
% Fm = fftshift(fft2(pyrMask));
% Ws = ones(1,nTheta)/nTheta;
% Pup = zeros(fovInPixel);
% [n1,n2,n3] = size(pupil);
% u = fix(2+nPxPup*(2*Samp-1)/2:nPxPup*(2*Samp+1)/2+1);
% Pup(u,u,:) = reshape(pupil, n1,[],1);
% 
% 
% x = reshape(modes(:,1),[nPxPup nPxPup]);
% phi_i = zeros(fovInPixel);
% phi_i(u,u,:) = reshape(x, n1,[],1);
% 
% 
% Imq = zeros(fovInPixel);
% for k = 1:nTheta
% f1 = ifft2(fft2(Pup.*fftPhasor(:,:,k)).*fft2(Fm));
% f2 = ifft2(fft2(Pup.*fftPhasor(:,:,k).*phi_i).*fft2(Fm));
% Imq = Imq + (2*imag(f1.*conj(f2)))/nTheta;
% end
% norm(Imq,2)

%%
mv = create_pyMask(fovInPixel,rooftop,alpha);
m1 = mv.*exp(1j.*OL1_base);
m2 = mv.*exp(1j.*OL1_noise);
for m = 1:length(Mods)
modulation = Mods(m);
nTheta = round(2*pi*Samp*modulation);
fftPhasor = GetModPhasor(nPxPup,Samp,modulation);
for k = 1:length(jModes)
x = reshape(modes(:,k),[nPxPup nPxPup]);
x1 = x/norm(x,2);
Sv(m,k) = Sensitivity(x1,nTheta,fftPhasor,mv,Samp,nPxPup,pupil);
S1(m,k) = Sensitivity(x1,nTheta,fftPhasor,m1,Samp,nPxPup,pupil);
S2(m,k) = Sensitivity(x1,nTheta,fftPhasor,m2,Samp,nPxPup,pupil);

x2 = x/norm(x,2);
Dv(m,k) = Linearity(x2,nTheta,fftPhasor,mv,Samp,nPxPup,pupil);
D1(m,k) = Linearity(x2,nTheta,fftPhasor,m1,Samp,nPxPup,pupil);
D2(m,k) = Linearity(x2,nTheta,fftPhasor,m2,Samp,nPxPup,pupil);


SDv(m,k) = SDFactor(Sv(m,k),Dv(m,k),1.3);
SD1(m,k) = SDFactor(S1(m,k),D1(m,k),1.3);
SD2(m,k) = SDFactor(S2(m,k),D2(m,k),1.3);

end

end
Dv = Dv.^-1;
D1 = D1.^-1;
D2 = D2.^-1;
%%

lbltxt{1} = sprintf("Pyr Mod $= %i\\lambda/D_0$   ",Mods(1));
lbltxt{2} = sprintf("Pyr Mod $= %i\\lambda/D_0$   ",Mods(2));
lbltxt{3} = sprintf("Pyr Mod $= %i\\lambda/D_0$   ",Mods(3));
lbltxt{4} = sprintf("Pyr+DE Mod $= %i\\lambda/D_0$",Mods(1));
lbltxt{5} = sprintf("Pyr+DE Mod $= %i\\lambda/D_0$",Mods(2));
lbltxt{6} = sprintf("Pyr+DE Mod $= %i\\lambda/D_0$",Mods(3));

% 
% 
% fig = figure('Color','w');
% hold on
% plot(1:length(jModes),Sv(1,:),'--xr')
% plot(1:length(jModes),Sv(2,:),'--xg')
% plot(1:length(jModes),Sv(3,:),'--xb')
% plot(1:length(jModes),S1(1,:),'-xr')
% plot(1:length(jModes),S1(2,:),'-xg')
% plot(1:length(jModes),S1(3,:),'-xb')
% set(gca,'XScale','log','YScale','log')
% legend(lbltxt,'interpreter','latex')
% 
% 
% fig = figure('Color','w');
% hold on
% plot(1:length(jModes),Dv(1,:),'--xr')
% plot(1:length(jModes),Dv(2,:),'--xg')
% plot(1:length(jModes),Dv(3,:),'--xb')
% plot(1:length(jModes),D1(1,:),'-xr')
% plot(1:length(jModes),D1(2,:),'-xg')
% plot(1:length(jModes),D1(3,:),'-xb')
% set(gca,'XScale','log','YScale','log')
% legend(lbltxt,'interpreter','latex')
% 
% fig = figure('Color','w');
% hold on
% plot(1:length(jModes),Sv(1,:).*Dv(1,:),'--xr')
% plot(1:length(jModes),Sv(2,:).*Dv(2,:),'--xg')
% plot(1:length(jModes),Sv(3,:).*Dv(3,:),'--xb')
% plot(1:length(jModes),S1(1,:).*D1(1,:),'-xr')
% plot(1:length(jModes),S1(2,:).*D1(2,:),'-xg')
% plot(1:length(jModes),S1(3,:).*D1(3,:),'-xb')
% set(gca,'XScale','log','YScale','log')
% legend(lbltxt,'interpreter','latex')
%% Fullplot
ylimit = sort([floor(min(min([Dv(:) D1(:) D2(:)]))) ceil(max(max([Sv(:) S1(:) S2(:)])))+5 ]);
% ylimit = [1e-2 1e2]

lw = 1;
fig = figure('Color','w','Position',[670 348 1065 624]);
subplot(3,2,[1 3])
hold on
plot(1:length(jModes),Sv(1,:),'--r','LineWidth',lw)
plot(1:length(jModes),Sv(2,:),'--g','LineWidth',lw)
plot(1:length(jModes),Sv(3,:),'--b','LineWidth',lw)
plot(1:length(jModes),S1(1,:),'-r','LineWidth',lw)
plot(1:length(jModes),S1(2,:),'-g','LineWidth',lw)
plot(1:length(jModes),S1(3,:),'-b','LineWidth',lw)

plot(1:length(jModes),SDv(1,:),'--r','LineWidth',lw)
plot(1:length(jModes),SDv(2,:),'--g','LineWidth',lw)
plot(1:length(jModes),SDv(3,:),'--b','LineWidth',lw)
plot(1:length(jModes),SD1(1,:),'-r','LineWidth',lw)
plot(1:length(jModes),SD1(2,:),'-g','LineWidth',lw)
plot(1:length(jModes),SD1(3,:),'-b','LineWidth',lw)

plot(1:length(jModes),Dv(1,:),'--r','LineWidth',lw)
plot(1:length(jModes),Dv(2,:),'--g','LineWidth',lw)
plot(1:length(jModes),Dv(3,:),'--b','LineWidth',lw)
plot(1:length(jModes),D1(1,:),'-r','LineWidth',lw)
plot(1:length(jModes),D1(2,:),'-g','LineWidth',lw)
plot(1:length(jModes),D1(3,:),'-b','LineWidth',lw)
xlabel("Zernike radial order",'interpreter','latex')
set(gca,'XScale','log','YScale','log','FontSize',20,'TickLabelInterpreter','latex','LineWidth',1)
box on
% ylim(ylimit)

subplot(3,2,[1 3]+1)
hold on
plot(1:length(jModes),Sv(1,:),'--r','LineWidth',lw)
plot(1:length(jModes),Sv(2,:),'--g','LineWidth',lw)
plot(1:length(jModes),Sv(3,:),'--b','LineWidth',lw)
plot(1:length(jModes),S2(1,:),'-r','LineWidth',lw)
plot(1:length(jModes),S2(2,:),'-g','LineWidth',lw)
plot(1:length(jModes),S2(3,:),'-b','LineWidth',lw)

plot(1:length(jModes),SDv(1,:),'--r','LineWidth',lw)
plot(1:length(jModes),SDv(2,:),'--g','LineWidth',lw)
plot(1:length(jModes),SDv(3,:),'--b','LineWidth',lw)
plot(1:length(jModes),SD2(1,:),'-r','LineWidth',lw)
plot(1:length(jModes),SD2(2,:),'-g','LineWidth',lw)
plot(1:length(jModes),SD2(3,:),'-b','LineWidth',lw)

plot(1:length(jModes),Dv(1,:),'--r','LineWidth',lw)
plot(1:length(jModes),Dv(2,:),'--g','LineWidth',lw)
plot(1:length(jModes),Dv(3,:),'--b','LineWidth',lw)
plot(1:length(jModes),D2(1,:),'-r','LineWidth',lw)
plot(1:length(jModes),D2(2,:),'-g','LineWidth',lw)
plot(1:length(jModes),D2(3,:),'-b','LineWidth',lw)
xlabel("Zernike radial order",'interpreter','latex')
set(gca,'XScale','log','YScale','log','FontSize',20,'TickLabelInterpreter','latex','LineWidth',1)
box on
%ylim(ylimit)
leg = legend(lbltxt,'FontSize',14,'interpreter','latex','NumColumns',2,'Position',[0.3225 0.1583 0.3838 0.1046]);


fold = "./figures/";
name = "SD_Performance.pdf";
exportgraphics(fig,fold+name)
%% FUNCTIONS
unwraper = @(x) unwrap(unwrap(x,[],2),[],1);

function s = Sensitivity(x,nTheta,fftPhasor,pyrMask,Samp,nPxPup,pupil)
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



function d = Linearity(x,nTheta,fftPhasor,pyrMask,Samp,nPxPup,pupil)
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