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
physicalParams.ReadoutNoise         = 0.5;
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

%% Meas
physicalParams.r0            = 3;
ReadoutNoise = 0.;
PhotonNoise = 0;
nPhotonBackground = 0;
quantumEfficiency = 1;
rng(rand)
atm = GenerateAtmosphereParameters(physicalParams);atm.idx

X_phase = zeros(560,560,100);
Y_z     = zeros(100,65);


for k = 1:100
[x,Zg] = ComputePhaseScreen(atm,PhaseCM);


end