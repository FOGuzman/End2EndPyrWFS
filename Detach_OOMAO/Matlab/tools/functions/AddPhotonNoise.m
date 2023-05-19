function [y] = AddPhotonNoise(y,nPhotonBackground,quantumEfficiency)
buffer    = y + nPhotonBackground;
y = y + randn(size(y)).*(y + nPhotonBackground);
index     = y<0;
y(index) = buffer(index);
y = quantumEfficiency*y;
end

