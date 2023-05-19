function [phaseMap,Zg] = ComputePhaseScreen(atm,CM)
N = atm.N;

phaseMap = real(ifft2(atm.idx.*sqrt(fftshift(atm.psdAO_mean)).*fft2(randn(N))./N).*atm.fourierSampling).*N.^2;
phaseMap = atm.pupil.*phaseMap(1:atm.nPxPup,1:atm.nPxPup);

Zg = CM*phaseMap(:);

end