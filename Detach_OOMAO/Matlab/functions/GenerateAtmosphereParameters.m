function [atm] = GenerateAtmosphereParameters(nLenslet,D,binning,r0,L0,fR0,modulation,fovInPixel,resAO,Samp,nPxPup,pupil)
noiseVariance = 0.7;
n_lvl         = 0.2;             % noise level in rad^2

nTimes        = fovInPixel/resAO;

N             = 2*Samp*nPxPup;
L             = (N-1)*D/(nPxPup-1);

fc = 1/binning*0.5*(nLenslet)/D;


[fx,fy] = freqspace(resAO,'meshgrid');
fx = fx*fc + 1e-7;
fy = fy*fc + 1e-7;

[Rx, Ry] =  RxRy(fx,fy,fc,nLenslet+1,D,r0,L0,fR0,modulation,binning,noiseVariance);


psdFit = fittingPSD(fx,fy,fc,'square',nTimes,r0,L0,fR0,D)./r0^(-5/3);
psdNoise = noisePSD(fx,fy,fc,Rx,Ry,noiseVariance,D)/noiseVariance;


[fxExt,fyExt] = freqspace(size(fx,1)*nTimes,'meshgrid');
fxExt = fxExt*fc*nTimes;
fyExt = fyExt*fc*nTimes;
index = abs(fxExt)<fc & abs(fyExt)<fc;

[SxAv,SyAv] = SxyAv(fx,fy,D,nLenslet);

psdAO_mean = zeros(size(fxExt));
aSlPSD = anisoServoLagPSD(fx,fy,fc,Rx,Ry,SxAv,SyAv,r0,L0,fR0,D);
psdAO_mean(index) = aSlPSD  + psdNoise*n_lvl;
psdAO_mean = psdAO_mean + psdFit*r0^(-5/3);

fourierSampling = 1./L;

[fx,fy] = freqspace(N,'meshgrid');
[~,fr]  = cart2pol(fx,fy);
fr      = fftshift(fr.*(N-1)/L./2);
[idx]           = find(fr==0);



atm.idx = idx;
atm.psdAO_mean = psdAO_mean;
atm.fourierSampling = fourierSampling;
atm.N = N;
atm.nPxPup = nPxPup;
atm.pupil = pupil;

end