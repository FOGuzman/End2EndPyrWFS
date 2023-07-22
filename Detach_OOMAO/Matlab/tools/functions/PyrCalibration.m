function [CM,I_0,IM] = PyrCalibration(params,PreC,flag)
flatMode = params.flatMode;
nPxPup = params.nPxPup;
modes = params.modes;
jModes = params.jModes;
Mul    = params.Multiplex;
DivMul = length(jModes)/Mul;
ZStack = zeros(length(jModes),length(jModes));

for m = 1:length(jModes)
  in = [1:DivMul:length(jModes)]+(m-1);
  for ch = 1:length(in)
  if in(ch) > length(jModes)
      in(ch) = in(ch)- length(jModes);
  end
  end
  ZStack(m,in) = ones(1,length(in));
end

IM = [];

I_0 = PropagatePyr(params,reshape(flatMode,[nPxPup nPxPup]),PreC,flag);
I_0 = I_0/sum(I_0,'all');
amp = .25;

if Mul > 1
   modes = modes*ZStack;
end

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

%if Mul > 1
%   IM = IM*ZStack;
%end

CM = pinv(IM);
end

