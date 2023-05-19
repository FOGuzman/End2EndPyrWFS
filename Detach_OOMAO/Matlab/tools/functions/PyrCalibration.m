function [CM,I_0,IM] = PyrCalibration(params,PreC,flag)
flatMode = params.flatMode;
nPxPup = params.nPxPup;
modes = params.modes;
jModes = params.jModes;

IM = [];
       
I_0 = PropagatePyr(params,reshape(flatMode,[nPxPup nPxPup]),PreC,flag);
I_0 = I_0/sum(I_0,'all');
amp = 1;

for k = 1:length(jModes)
imMode = reshape(modes(:,k),[nPxPup nPxPup]);
%push
z = imMode.*amp;
I4Q = PropagatePyr(params,z,PreC,flag);
sp = I4Q/sum(I4Q,'all')-I_0;

%pull
I4Q = PropagatePyr(params,-z,PreC,flag);
sm = I4Q/sum(I4Q,'all')-I_0;

MZc = 0.5*(sp-sm)/amp;
IM = [IM MZc(:)];
end
CM = pinv(IM);
end

