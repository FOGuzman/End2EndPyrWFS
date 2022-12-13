clear all; close all
addpath functions
oomao_path = "/home/fg/Desktop/OOMAO/Vanilla_OOMAO/Simulations/oomao/OOMAO-master/OOMAO-master";
%preFold = "../Preconditioners/nocap/pnoise/checkpoint/OL1_R128_M0_RMSE0.05275_Epoch_118.mat";
preFold = "../Preconditioners/nocap/base/checkpoint/OL1_R128_M0_RMSE0.02807_Epoch_91.mat";

addpath(genpath(oomao_path))


binning       = 1;
D             = 8;
modulation    = 1;
nLenslet      = 16;
resAO         = 2*nLenslet+1;
L0            = 30;
r0            = 0.7;
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

jModes = [2:200];

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
        
%% Loop parameters
numIter = 1000;
gain = 0.9;
stroke = 10;

cmos.resolution = 128;
cmos.nyquistSampling = 16;
cmos.fieldStopSize = 10;

ReadoutNoise = 0.;
PhotonNoise = 0;
nPhotonBackground = 0;
quantumEfficiency = 1;

ref_psf = DetachImager(cmos,reshape(flatMode,[nPxPup nPxPup]));
%% OOMAO parameters
Alts = [4,10]*1e3;
FR0s = [0.7,0.3];
WS = [1,2];
WD = [0,pi/4];
s = RandStream('mt19937ar','Seed',15);
ngs = source('wavelength',photometry.R);
atm = atmosphere(photometry.R,r0,L0,'altitude',Alts,...
    'fractionnalR0',FR0s,...
    'windSpeed',WS,...
    'randStream',s,...
    'windDirection',WD);

wvlf = atm.wavelength/(2*pi)/1e-6;
tel = telescope(D,...
    'fieldOfViewInArcMin',30,...
    'resolution',nPxPup,...
    'samplingTime',1/300);
tel = tel + atm;
ngs = ngs.*tel
+tel
%% Loop
% preconf fig
phi_buffer1 = reshape(flatMode,[nPxPup nPxPup])*0;
phi_buffer2 = reshape(flatMode,[nPxPup nPxPup])*0;
we_openloop = 0;
we1 = 0;
we2 = 0;
midL = round(size(ref_psf,1)/2);


fig = figure('Color','w','Position',[1 42 1920 957]);

subplot(3,5,1)
ha_phi.img = imagesc(phi_buffer1,'AlphaData',pupil);axis off;colormap jet;axis image
ha_phi.cb = colorbar();
ha_phi.title = title("$\phi_i^t$",'interpreter','latex','FontSize',16);

subplot(3,5,2)
ha_phi1a.img = imagesc(phi_buffer1,'AlphaData',pupil);axis off;colormap jet;axis image
ha_phi1a.cb = colorbar();
ha_phi1a.title = title("$\phi_i^t + \phi_{pyr}^{t-1}$",'interpreter','latex','FontSize',16);

subplot(3,5,3)
ha_phi1b.img = imagesc(phi_buffer1,'AlphaData',pupil);axis off;colormap jet;axis image
ha_phi1b.cb = colorbar();
ha_phi1b.title = title("$\phi_{pyr}^{t}$",'interpreter','latex','FontSize',16);

subplot(3,5,4)
ha_sc1.img = imagesc(ref_psf);colormap jet;axis image
ha_sc1.title = title("$SC1$",'interpreter','latex','FontSize',16);
set(gca,'FontSize',14,'TickLabelInterpreter','latex')

subplot(3,5,5)
ha_psf1.ref = plot(ref_psf(midL,:),'-k','LineWidth',2);hold on
ha_psf1.pyr = plot(ref_psf(midL,:),'-r','LineWidth',2);box on;grid on
xlim([1 size(ref_psf,1)])
set(gca,'FontSize',14,'TickLabelInterpreter','latex')
legend("Ideal","Pyr",'interpreter','latex','FontSize',8)

subplot(3,5,7)
ha_phi2a.img = imagesc(phi_buffer1,'AlphaData',pupil);axis off;colormap jet;axis image
ha_phi2a.cb = colorbar();
ha_phi2a.title = title("$\phi_i^t + \phi_{pyr+DE}^{t-1}$",'interpreter','latex','FontSize',16);

subplot(3,5,8)
ha_phi2b.img = imagesc(phi_buffer1,'AlphaData',pupil);axis off;colormap jet;axis image
ha_phi2b.cb = colorbar();
ha_phi2b.title = title("$\phi_{pyr+DE}^{t}$",'interpreter','latex','FontSize',16);


subplot(3,5,9)
ha_sc2.img = imagesc(ref_psf);colormap jet;axis image
ha_sc2.title = title("$SC2$",'interpreter','latex','FontSize',16);
set(gca,'FontSize',14,'TickLabelInterpreter','latex')

subplot(3,5,10)
ha_psf2.ref = plot(ref_psf(midL,:),'-k','LineWidth',2);hold on
ha_psf2.de = plot(ref_psf(midL,:),'-b','LineWidth',2);box on;grid on
xlim([1 size(ref_psf,1)])
set(gca,'FontSize',14,'TickLabelInterpreter','latex')
legend("Ideal","Pyr+DE",'interpreter','latex','FontSize',8)

subplot(3,5,[11 15])
ha_line.p1 = plot(1,we_openloop,'-k','LineWidth',2);hold on
ha_line.p2 = plot(1,we1,'-r','LineWidth',2);
ha_line.p3 = plot(1,we2,'-b','LineWidth',2);
xlim([1 numIter]);box on;grid on;
ylabel("RMSE",'interpreter','latex','FontSize',13)
xlabel("$n^{\circ}$ Iteration",'interpreter','latex','FontSize',13)
legend("Open loop","Pyr","Pyr+DE",'interpreter','latex','FontSize',13)
set(gca,'FontSize',14,'TickLabelInterpreter','latex')


str={'Atmosphere parameters:',...
    "Von Karman atmospheric turbulence",...   
    sprintf("$$\\lambda = %.2f [\\mu m]$$",atm.wavelength*1e6),...
    sprintf("$$r_0 = %.2f [cm]$$",atm.r0*100),...
    sprintf("$$D= %.2f [m]$$",D),...
    sprintf("$$L_0 = %.2f [m]$$",atm.L0),...
    sprintf("seeing $$ = %.2f [arcsec]$$",atm.seeingInArcsec),...
    sprintf("$$\\theta_0 = %.2f [arcsec]$$",atm.theta0InArcsec),...
    sprintf("$$\\tau_0 = %.2f [ms]$$",atm.tau0InMs),...
    sprintf("Layers $$ = %i$$",atm.nLayer),...
    sprintf("Altitude $$ = [" + repmat('\\ %.0f',1,atm.nLayer)+ "] [m]$$",Alts),...
    sprintf("Fractional $$r_0 = [" + repmat('\\ %.2f',1,atm.nLayer)+ "]$$",FR0s),...
    sprintf("Wind speed $$ = [" + repmat('\\ %.0f',1,atm.nLayer)+ "]$$",WS),...
    sprintf("Wind direction $$ = [" + repmat('\\ %.0f',1,atm.nLayer)+ "][^{\\circ}]$$",WD*180/pi),...
    };
Tx = annotation('textbox','interpreter'...
    ,'latex','String',str,'FitBoxToText','on');
set(Tx,'Position',[0.12 0.38 0.154 0.3],'FontSize',12)
%%
phi_buffer1 = reshape(flatMode,[nPxPup nPxPup])*0;
phi_buffer2 = reshape(flatMode,[nPxPup nPxPup])*0;
phi_res1 = phi_buffer1;
phi_res2 = phi_res1;
we_openloop = [];
we1 = [];
we2 = [];
itx = [];

for k = 1:numIter
+tel;
+ngs;
phi =    ngs.phase;

%props
y = PropagatePyr(fovInPixel,phi_res1,Samp,modulation,rooftop,alpha,pupil,nPxPup,OL1_trained,0);
if PhotonNoise
y = AddPhotonNoise(y,nPhotonBackground,quantumEfficiency);
end
y = y +randn(size(y)).*ReadoutNoise;
Pyr_y = y/sum(y(:))-PyrI_0;
% Estimation
PyrZe = PyrCM*Pyr_y(:);
phi_hat1 = reshape(modes*PyrZe*gain,[nPxPup nPxPup]);

y = PropagatePyr(fovInPixel,phi_res2,Samp,modulation,rooftop,alpha,pupil,nPxPup,OL1_trained,1);
if PhotonNoise
y = AddPhotonNoise(y,nPhotonBackground,quantumEfficiency);
end
y = y +randn(size(y)).*ReadoutNoise;
Net_y = y/sum(y(:))-NetI_0;
% Estimation
NetZe = NetCM*Net_y(:);
phi_hat2 = reshape(modes*NetZe*gain,[nPxPup nPxPup]);


phi_res1 = phi+phi_buffer1 - phi_hat1;
phi_res2 = phi+phi_buffer1 - phi_hat2;

phi_buffer1 = -phi_hat1;
phi_buffer2 = -phi_hat2;

sc1 = DetachImager(cmos,phi_res1);
sc2 = DetachImager(cmos,phi_res2);

we_openloop = [we_openloop sqrt(mse(phi(pupil==1)))];
we1 = [we1 sqrt(mse(phi_res1(pupil==1)))];
we2 = [we2 sqrt(mse(phi_res2(pupil==1)))];
itx = [itx k];

%update fig
ha_phi.img.CData = phi;
ha_phi1a.img.CData = phi_res1;
ha_phi1b.img.CData = phi_hat1;
ha_sc1.img.CData = sc1;
ha_phi2a.img.CData = phi_res2;
ha_phi2b.img.CData = phi_hat2;
ha_sc2.img.CData = sc2;

ha_psf1.pyr.YData = sc1(midL,:);
ha_psf2.de.YData = sc2(midL,:);

ha_line.p1.XData = itx; ha_line.p1.YData = we_openloop;
ha_line.p2.XData = itx; ha_line.p2.YData = we1;
ha_line.p3.XData = itx; ha_line.p3.YData = we2;

drawnow
end



% cmos.resolution = 124;
% cmos.nyquistSampling = 16;
% cmos.fieldStopSize = 20;
% sc1 = DetachImager(cmos,phi_res1);
% imagesc(sc1)