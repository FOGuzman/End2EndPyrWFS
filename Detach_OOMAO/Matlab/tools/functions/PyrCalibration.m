function [CM,I_0,IM] = PyrCalibration(params,PreC,flag)
flatMode = params.flatMode;
nPxPup = params.nPxPup;
modes = params.modes;
jModes = params.jModes;
Mul    = params.Multiplex;
DivMul = length(jModes)/Mul;
ZStack = zeros(DivMul,length([1:DivMul:length(jModes)]));
for m = 1:DivMul
  ZStack(m,:) = [1:DivMul:length(jModes)]+(m-1);
end

IM = [];

I_0 = PropagatePyr(params,reshape(flatMode,[nPxPup nPxPup]),PreC,flag);
I_0 = I_0/sum(I_0,'all');
amp = .1;
if Mul == 1
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
else
    for kz = 1:size(ZStack,1)
       imMode = reshape(modes(:,ZStack(kz,:)),[nPxPup nPxPup, Mul]);
       zm     = sum(imMode,3);
       z = zm.*amp;
       I4Q = PropagatePyr(params,z,PreC,flag);
       sp = I4Q/sum(I4Q,'all')-I_0;
       
       %pull
       I4Q = PropagatePyr(params,-z,PreC,flag);
       sm = I4Q/sum(I4Q,'all')-I_0;
        
       MZc = 0.5*(sp-sm)/amp;
       IM = [IM MZc(:)];
    end
end
CM = pinv(IM);
end

