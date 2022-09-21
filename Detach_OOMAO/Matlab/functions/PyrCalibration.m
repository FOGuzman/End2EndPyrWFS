function [CM,I_0] = PyrCalibration(jModes,modes,flatMode,fovInPixel,nPxPup,...
    Samp,modulation,rooftop,alpha,pupil,PreC,flag)
IM = [];
       
I_0 = PropagatePyr(fovInPixel,reshape(flatMode,[nPxPup nPxPup])...
    ,Samp,modulation,rooftop,alpha,pupil,nPxPup,PreC,flag);
I_0 = I_0/sum(I_0,'all');
amp = 1;

for k = 1:length(jModes)
imMode = reshape(modes(:,k),[nPxPup nPxPup]);
%push
z = imMode.*amp;
I4Q = PropagatePyr(fovInPixel,z,Samp,modulation,rooftop,alpha,pupil,nPxPup,PreC,flag);
sp = I4Q/sum(I4Q,'all')-I_0;

%pull
I4Q = PropagatePyr(fovInPixel,-z,Samp,modulation,rooftop,alpha,pupil,nPxPup,PreC,flag);
sm = I4Q/sum(I4Q,'all')-I_0;

MZc = 0.5*(sp-sm)/amp;
IM = [IM MZc(:)];
end
CM = pinv(IM);
end

