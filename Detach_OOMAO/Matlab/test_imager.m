
addpath functions
clear all

preFold = "../Preconditioners/nocap/pnoise/checkpoint/OL1_R128_M0_RMSE0.05275_Epoch_118.mat";

binning       = 1;
D             = 1.5;
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

jModes = [2:64];

%% Pyramid calibration
modes = CreateZernikePolynomials(nPxPup,jModes,pupil~=0);
flatMode = CreateZernikePolynomials(nPxPup,1,pupil~=0);


%% Test
% Phase inversion
PhaseCM = pinv(modes);

load(preFold);OL1_trained = OL1;
%OL1_trained = 20*rand(512,512);

[NetCM,NetI_0] = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,Samp...
    ,modulation,rooftop,alpha,pupil,OL1_trained,1);

[PyrCM,PyrI_0,PyrIM] = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,Samp...
    ,modulation,rooftop,alpha,pupil,OL1_trained,0);
        
%% Meas
r0            = 0.1;
ReadoutNoise = 0.;
PhotonNoise = 0;
nPhotonBackground = 0;
quantumEfficiency = 1;
atm = GenerateAtmosphereParameters(nLenslet,D,binning,r0,L0,fR0,modulation,fovInPixel,resAO,Samp,nPxPup,pupil);

cmos.resolution = 128;
cmos.nyquistSampling = 16;
cmos.fieldStopSize = 20;

fig = figure;
subplot(1,2,1)
im1 = imagesc(zeros(nPxPup));
subplot(1,2,2)
im2 = imagesc(zeros(cmos.resolution));


for k = 1:1000
[x,Zg] = ComputePhaseScreen(atm,PhaseCM);
mI = DetachImager(cmos,x);
set(im1,'CData',x)
set(im2,'CData',mI)
drawnow
end

imagesc(out)

function out = DetachImager(obj,waveIn)
%% PROPAGATETHROUGH Fraunhoffer wave propagation to the lenslets focal plane
%
% propagateThrough(obj) progates the object wave throught the
% lenslet array and computes the imagelets

% for a given wave of size [nxn], 
% nyquistSampling=1 means 2 pixels per fhwm obtained by setting
% fftPad=2, fieldStopSize should be set to default n/nLenslet
% fwhm
% nyquistSampling=0.5 means 1 pixels per fhwm obtained by
% setting fftPad=1, fieldStopSize should be set to default
% n/nLenslet/nyquistSampling/2
resolution = obj.resolution;
nyquistSampling = obj.nyquistSampling;
fieldStopSize = obj.fieldStopSize;
% Check for LGS asterisms
waveIn = exp(1j.*waveIn);
[n1,n2,n3] = size(waveIn);
n12 = n1*n2;
src = waveIn;

val = waveIn; % get complex amplitudes
[nLensletsWavePx,nLensletsWavePxNGuideStar,nWave] = size(val);
% Resize the 3D input into a 2D input
nLensletsWavePxNGuideStar = nLensletsWavePxNGuideStar*nWave;
val = reshape(val,nLensletsWavePx,nLensletsWavePxNGuideStar);
nLensletWavePx = nLensletsWavePx;
nLensletArray = nLensletsWavePxNGuideStar/nLensletsWavePx;
%             obj.nArray = nLensletArray;
% Invocation of the zoom optics for conjugation to finite
% distance
%             if isfinite(obj.conjugationAltitude)
% %                 val                 = obj.zoomTransmittance.*val;
%                 val = repmat(obj.zoomTransmittance,1,nLensletArray).*val;
%             end
%             nOutWavePx    = obj.nLensletImagePx*obj.fftPad;    % Pixel length of the output wave
%             evenOdd       = rem(obj.nLensletImagePx,2);
fftPad = ceil(nyquistSampling*2);
nOutWavePx    = nLensletWavePx*fftPad;    % Pixel length of the output wave
evenOdd       = rem(nLensletWavePx,2);
nOutWavePx = max(nOutWavePx,nLensletWavePx);
nLensletSquareWavePx    = nLensletWavePx;
wavePrgted = zeros(nOutWavePx,nLensletSquareWavePx);
val        = val./nOutWavePx;

nLensletImagePx = ceil(obj.fieldStopSize.*obj.nyquistSampling*2);

%             nLensletWavePx   = obj.nLensletWavePx;
%%% ODD # OF PIXELS PER LENSLET
%                 fprintf('ODD # OF PIXELS PER LENSLET (%d) Phasor empty!\n',obj.nLensletImagePx)
% Shape the wave per columns of lenslet pixels
val       = reshape(val,nLensletWavePx,nLensletSquareWavePx);
u         = any(val); % Index of non-zeros columns
wavePrgted(:,u) = fftshift(fft(val(:,u),nOutWavePx),1);

nLensletImagePx  = nLensletImagePx;
nLensletsImagePx = nLensletImagePx;
% Select the field of view
v = [];
if nOutWavePx>nLensletImagePx
%                     disp('Cropping!')
%                     centerIndex = (nOutWavePx+1)/2;
%                     halfLength  = (nLensletImagePx-1)/2;
    centerIndex = ceil((nOutWavePx+1)/2);
    halfLength  = floor(nLensletImagePx/2);
    v           = true(nOutWavePx,1);
%                     v((-halfLength:halfLength)+centerIndex) ...
%                                 = false;
    v((0:nLensletImagePx-1)-halfLength+centerIndex) ...
                = false;
elseif nOutWavePx<nLensletImagePx
    error('lensletArray:propagateThrough:size','The computed image is smaller than the expected image!')
end
wavePrgted(v,:) = [];
% Back to transpose 2D
val       = reshape( wavePrgted ,...
    nLensletsImagePx,nLensletWavePx*nLensletArray).';
% Shape the wave per rows of lenslet pixels
val       = reshape(val,nLensletWavePx,nLensletsImagePx*nLensletArray);
u         = any(val); % Index of non-zeros columns
wavePrgted = zeros(nOutWavePx,nLensletsImagePx*nLensletArray);
wavePrgted(:,u)  = fftshift(fft(val(:,u),nOutWavePx),1);
wavePrgted(v,:) = [];

% Back to transpose 2D
wavePrgted  = reshape(wavePrgted,nLensletsImagePx*nLensletArray,nLensletsImagePx).';
%             wavePrgted = wavePrgted.*conj(wavePrgted);
% and back to input wave array shape
[n,m] = size(wavePrgted);
wavePrgted = reshape(wavePrgted,[n,m/nWave,nWave]);
out = real(wavePrgted.*conj(wavePrgted));
out = imresize(out, [resolution resolution],'bilinear');
end