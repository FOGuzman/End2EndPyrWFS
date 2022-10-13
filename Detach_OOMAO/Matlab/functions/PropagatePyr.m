function [I4Q] = PropagatePyr(fovInPixel,phaseMap,Samp,modulation,rooftop,alpha,pupil,nPxPup,PreC,flag)



[uu,vv]       = ndgrid((0:(fovInPixel-1))./fovInPixel);
[o,r]         = cart2pol(uu,vv);
pyrPupil      = pupil.*exp(1i.*phaseMap);
u = fix(2+nPxPup*(2*Samp-1)/2:nPxPup*(2*Samp+1)/2+1);

nTheta = round(2*pi*Samp*modulation);
OL1 = exp(1j.*PreC);

[pyrMask] = create_pyMask(fovInPixel,rooftop,alpha);
PyrQ = zeros(fovInPixel);
I4Q4 = PyrQ;
Wpupil = pyrPupil;
[n1,n2,n3] = size(Wpupil);
PyrQ(u,u,:) = reshape(Wpupil, n1,[],1);
if nTheta > 0
    for kTheta = 1:nTheta
        theta = (kTheta-1)*2*pi/nTheta;
        fftPhasor = exp(-1i.*pi.*4*modulation*Samp*r.*cos(o+theta));
        buf = PyrQ.*fftPhasor; % atmosfera + fase mod
        if flag
            buf = fft2(buf).*pyrMask.*OL1; % P(atm+fas_mod)*fase_pyr*capa_entrenamble 
        else
            buf = fft2(buf).*pyrMask;
        end
        psf = abs(fft2(buf)).^2; % P(P(atm+fas_mod)*fase_pyr)
        I4Q4(:,:,:) = I4Q4(:,:,:) + abs(fft2(buf)).^2;
    end
    I4Q = I4Q4/nTheta;
else
    if flag == 1
        buf = fft2(PyrQ).*pyrMask.*OL1; % P(atm+fas_mod)*fase_pyr*capa_entrenamble 
    elseif flag == 2
        buf = fft2(PyrQ).*OL1; % P(atm+fas_mod)*fase_pyr*capa_entrenamble 
    else
        buf = fft2(PyrQ).*pyrMask;
    end
    I4Q = abs(fft2(buf)).^2;
end
I4Q = imresize(I4Q,1/(2*Samp))*2*Samp;
end

