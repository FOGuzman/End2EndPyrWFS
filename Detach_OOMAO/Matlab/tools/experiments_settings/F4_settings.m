physicalParams = struct();

% Pyramid propeties
physicalParams.D                    = 8;             % Telescope diameter [m]
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
physicalParams.jModes               = 2:60;

%Camera parameters
physicalParams.ReadoutNoise         = 0;
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