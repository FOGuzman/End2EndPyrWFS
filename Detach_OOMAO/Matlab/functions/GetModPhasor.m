function [fftPhasor] = GetModPhasor(nPxPup,Samp,modulation)
fovInPixel    = nPxPup*2*Samp;
[uu,vv]       = ndgrid((0:(fovInPixel-1))./fovInPixel);
[o,r]         = cart2pol(uu,vv);
u = fix(2+nPxPup*(2*Samp-1)/2:nPxPup*(2*Samp+1)/2+1);

nTheta = round(2*pi*Samp*modulation);
fftPhasor = zeros(fovInPixel,fovInPixel,nTheta);
if nTheta > 0
  for kTheta = 1:nTheta
        theta = (kTheta-1)*2*pi/nTheta;
        fftPhasor(:,:,kTheta) = exp(-1i.*pi.*4*modulation*Samp*r.*cos(o+theta));
  end
else
    fftPhasor = ones(fovInPixel,fovInPixel);
end
end

