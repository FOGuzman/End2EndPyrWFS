function [Z]=zernike_mask_vincent(nPx,n_sh,d,signe)

% Input : 
% nPx = size of the pupil if pixel exemple 128
% n_sh = sampling if n_sh = 2 we are at 2*Shannon meaning that the 
%matrix will be 4 times bigger than the pupil
%d = ;%DIAMTER IN L/D
% signe = +1 or -1 if you want a Positive mask or a negative one

%ZELDA
X = round(linspace(-n_sh*nPx,n_sh*nPx-1,2*n_sh*nPx));
[x,y] = meshgrid(X);
[theta,r]  = cart2pol(x,y);
Z = ones(2*n_sh*nPx);
Z(r<=(d/2)*2*n_sh) = 1j*signe;

end
