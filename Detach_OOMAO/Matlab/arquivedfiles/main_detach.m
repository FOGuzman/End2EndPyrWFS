addpath tools/functions
clear all

preFold = "D:\FOGuzman\End2EndPyrWFS\Detach_OOMAO\Preconditioners\n1_nico.mat";
%preFold = "/home/fg/Desktop/FOGuzman/End2EndPyrWFS/Detach_OOMAO/Pytorch/training_results/Paper/06-07-2023/n1_nico.mat";


physicalParams = struct();

% Pyramid propeties
physicalParams.modulation           = 0;
physicalParams.D                    = 3;             % Telescope diameter [m]
physicalParams.nLenslet             = 16;            % Equivalent SH resolution lenslet
physicalParams.binning              = 1;             % Binning in phase sampling
physicalParams.Samp                 = 2;             % Oversampling factor
physicalParams.nPxPup               = 128;           % number of pixels to describe the pupil
physicalParams.alpha                = pi/2;          % Pyramid shape
physicalParams.rooftop              = [0,0];         % Pyramid roftop imperfection
% Atmosphere propeties
physicalParams.L0                   = 25;            % Outer scale [m]
physicalParams.fR0                  = 1;             % Fracional r0 (for multi layer - not implemented)
% indecies for Zernike decomposition 
physicalParams.jModes               = 2:66;
physicalParams.Multiplex            = 1;
%Camera parameters
physicalParams.ReadoutNoise         = 0.0;
physicalParams.PhotonNoise          = 0;
physicalParams.quantumEfficiency    = 1;
physicalParams.nPhotonBackground    = 0;

% Precomp aditional parameters
physicalParams.resAO                = 2*physicalParams.nLenslet+1;
physicalParams.pupil                = CreatePupil(physicalParams.nPxPup,"disc");
physicalParams.N                    = 2*physicalParams.Samp*physicalParams.nPxPup;
physicalParams.L                    = (physicalParams.N-1)*physicalParams.D/(physicalParams.nPxPup-1);
physicalParams.fovInPixel           = physicalParams.nPxPup*2*physicalParams.Samp;    % number of pixel to describe the PSD
physicalParams.nTimes               = physicalParams.fovInPixel/physicalParams.resAO;
physicalParams.PyrQ                 = zeros(physicalParams.fovInPixel);
physicalParams.I4Q4                 = physicalParams.PyrQ;

physicalParams.modes = CreateZernikePolynomials(physicalParams.nPxPup,physicalParams.jModes,physicalParams.pupil~=0);
physicalParams.flatMode = CreateZernikePolynomials(physicalParams.nPxPup,1,physicalParams.pupil~=0);


%% Test
% Phase inversion
PhaseCM = pinv(physicalParams.modes);

load(preFold);OL1_trained = OL1;
%OL1_trained = 20*rand(512,512);

[DPWFS_CM,DPWFS_I0,DPWFS_IM] = PyrCalibration(physicalParams,OL1_trained,1);

[PyrCM,PyrI_0,PyrIM] = PyrCalibration(physicalParams,OL1_trained,0);
  
figure,
plot(diag(PyrIM'*PyrIM),'k')
hold on
plot(diag(DPWFS_IM'*DPWFS_IM),'r')

%% Meas
physicalParams.r0            = 3;
ReadoutNoise = 0.;
PhotonNoise = 0;
nPhotonBackground = 0;
quantumEfficiency = 1;
rng(rand)
atm = GenerateAtmosphereParameters(physicalParams);atm.idx
[x,Zg] = ComputePhaseScreen(atm,PhaseCM);
%%
%x_n = uint8(x/pi*255+128);
%x = (double(x_n)-128)*pi/255;
Zg = PhaseCM*x(:);
%Zg = Y_z(10,:)';
[Zpyr]    = PropAndSamp(physicalParams,x,OL1_trained ,PyrI_0   ,PyrCM    ,0);
[Zdpwfs,ydpwfs]  = PropAndSamp(physicalParams,x,OL1_trained ,DPWFS_I0 ,DPWFS_CM ,1);


imsc = [min(x(:)),max(x(:))];
figure('Position',[8 81 1886 874]),
subplot(341)
imagesc(x,[imsc(1) imsc(2)]);axis image;colorbar
title('Input Phase')
subplot(342)
imagesc(reshape(physicalParams.modes*Zg,[physicalParams.nPxPup physicalParams.nPxPup]));axis image;colorbar
title('Phase inversion')
subplot(343)
imagesc(reshape(physicalParams.modes*Zpyr,[physicalParams.nPxPup physicalParams.nPxPup]));axis image;colorbar
title('Pyramid estimation')
subplot(344)
imagesc(reshape(physicalParams.modes*Zdpwfs,[physicalParams.nPxPup physicalParams.nPxPup]));axis image;colorbar
title('Network estimation')
subplot(345)
imagesc(ydpwfs);axis image;colorbar
title('Network estimation')

res1 = x-reshape(physicalParams.modes*Zg,[physicalParams.nPxPup physicalParams.nPxPup]);
res2 = x-reshape(physicalParams.modes*Zpyr,[physicalParams.nPxPup physicalParams.nPxPup]);
res3 = x-reshape(physicalParams.modes*Zdpwfs,[physicalParams.nPxPup physicalParams.nPxPup]);
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
plot(Zpyr,'r','linewidth',2)
plot(Zdpwfs,'b','linewidth',2)
legend('Groundtruth','Traditional Pyramid','Pyramid with preconditioner')
grid on; box on
fprintf("RMSE Pyr: %.5f  | DE: %.5f\n",sqrt(mse(Zg,Zpyr)),sqrt(mse(Zg,Zdpwfs)))

%%
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

function [s] = mse(x,y)

s =mean (x.^2+y.^2);


end