function [SxAv,SyAv] = SxyAv(fx,fy,D,nLenslet)
d = D/(nLenslet);

Sx = 1i*2*pi*fx*d ;
Sy = 1i*2*pi*fy*d ;

Av = sinc(d*fx).*sinc(d*fy).*exp(1i*pi*d*(fx+fy));
SxAv = Sx.*Av;
SyAv = Sy.*Av;
end

