addpath functions
clear all

preFold = "../Preconditioners/OL1_R128_M0_RMSE0.04296_Epoch_99.mat";


jModes        = [2:36];
jModes = [0 2 4 8 12 18 24 32 40 50 60 72 84 98 112 128]+1;
mods = [0 1 3];
sVec = zeros(1,length(jModes));


binning       = 1;
D             = 8;
modulation    = 1;
nLenslet      = 16;
resAO         = 2*nLenslet+1;
r0            = 0.8;
L0            = 25;
fR0           = 1;
noiseVariance = 0.7;
n_lvl         = 0.1;             % noise level in rad^2
Samp          = 2;                % OVer-sampling factor
nPxPup        = 128;               % number of pixels to describe the pupil
alpha         = pi/2;
rooftop       = [0,0]; 
fovInPixel    = nPxPup*2*Samp;    % number of pixel to describe the PSD
pupil         = CreatePupil(nPxPup,"disc");

ReadoutNoise  = 0;


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

[CM,I0] = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,Samp...
    ,modulation,rooftop,alpha,pupil,OL1_trained,0);


%% IL

nTheta = round(2*pi*Samp*modulation);
[n1,n2,n3] = size(pyrPupil);
for k = 1:length(jModes)
pyrPupil      = pupil.*exp(1i.*reshape(flatMode,[nPxPup nPxPup]));
u = fix(2+nPxPup*(2*Samp-1)/2:nPxPup*(2*Samp+1)/2+1);
PyrQ = zeros(fovInPixel);

PyrQ(u,u,:) = reshape(pyrPupil, n1,[],1);



pyrPupil      = pupil.*exp(1i.*reshape(modes(:,k),[nPxPup nPxPup]));
u = fix(2+nPxPup*(2*Samp-1)/2:nPxPup*(2*Samp+1)/2+1);
phi_t = zeros(fovInPixel);
[n1,n2,n3] = size(pyrPupil);
phi_t(u,u,:) = angle(reshape(pyrPupil, n1,[],1));


m = create_pyMask(fovInPixel,rooftop,alpha);
A = conv2(PyrQ.*phi_t,fftshift(fft2(m)),'same');
Il = 2*imag(imresize(I0,2*Samp).*conj(A));
sVec(mm,k) = norm(Il,2);k
end


end

%%
hold on
loglog(1:length(jModes),sVec(1,:))
loglog(1:length(jModes),sVec(2,:))
loglog(1:length(jModes),sVec(3,:));xlim([1 22]);ylim([0.6e-2 10^3])