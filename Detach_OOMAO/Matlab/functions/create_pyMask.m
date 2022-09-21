function [pyrMask] = create_pyMask(pixSide,rooftop,alpha)
if isscalar(alpha)
    realAlpha = ones(1,4)*alpha;
else
    realAlpha = alpha;
end
% Complex alpha for PYR surface slope error in 2 directions...
imagAlpha = imag(realAlpha);
realAlpha = real(realAlpha);

nx = rooftop(1);
ny = rooftop(2);

[fx,fy] = freqspace(pixSide,'meshgrid');
fx = fx.*floor(pixSide/2);
fy = fy.*floor(pixSide/2);
%pym = zeros(pxSide);

% pyramid face transmitance and phase for fx>=0 & fy>=0
mask  = graduatedHeaviside(fx,nx).*graduatedHeaviside(fy,nx);
phase = -realAlpha(1).*(fx+fy) + -imagAlpha(1).*(-fx+fy);
pym   = mask.*exp(1i.*phase);
% pyramid face transmitance and phase for fx>=0 & fy<=0
mask  = graduatedHeaviside(fx,ny).*graduatedHeaviside(-fy,-ny);
phase = -realAlpha(2).*(fx-fy) + -imagAlpha(2).*(fx+fy);
pym   = pym + mask.*exp(1i.*phase);
% pyramid face transmitance and phase for fx<=0 & fy<=0
mask  = graduatedHeaviside(-fx,-nx).*graduatedHeaviside(-fy,-nx);
phase = realAlpha(3).*(fx+fy) + -imagAlpha(3).*(fx-fy);
pym   = pym + mask.*exp(1i.*phase);
% pyramid face transmitance and phase for fx<=0 & fy>=0
mask  = graduatedHeaviside(-fx,-ny).*graduatedHeaviside(fy,ny);
phase = -realAlpha(4).*(-fx+fy) + -imagAlpha(4).*(-fx-fy);
pym   = pym + mask.*exp(1i.*phase);
pyrMask   = fftshift(pym./sum(abs(pym(:))));
end

