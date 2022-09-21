function [fun] = CreateZernikePolynomials(nPxPup,jModes,pupilLogical)
u = nPxPup;
u = 2.*[-(u-1)/2:(u)/2]/u;
v = u;
[x,y] = meshgrid(u,v);
[o,r] = cart2pol(x,y) ;

mode = jModes;
nf = zeros(1,length(mode));
mf = zeros(1,length(mode));


for counti=1:size(mode,1)
    for countj=1:size(mode,2)

        %ordre radial
        n = 0;
        %ordre azimuthal
        m = 0;

        count = 0;

        while length(n)<mode(counti,countj)

            count = count + 1;
            tmp = 0:count;
            tmp = tmp(rem(count-tmp,2)==0);

            if all(tmp)
                n = [n ones(1,2.*length(tmp)).*count];
                tmp1 = tmp(ones(2,1),:);
                m = [m reshape(tmp1,1,size(tmp1,1).*size(tmp1,2))];
            else
                n = [n ones(1,(2.*length(tmp))-1).*count];
                tmp1 = tmp(ones(2,1),2:length(tmp));
                m = [m 0 reshape(tmp1,1,size(tmp1,1).*size(tmp1,2))];
            end

        end

        nf(counti,countj) = n(mode(counti,countj));
        mf(counti,countj) = m(mode(counti,countj));

    end
end
nv = nf;
mv = mf;
nf  = length(jModes);
fun = zeros(numel(r),nf);
r = r(pupilLogical);
o = o(pupilLogical);

 
ind_m = find(mv==0);
for cpt=ind_m
n = nv(cpt);
m = mv(cpt);
fun(pupilLogical,cpt) = sqrt(n+1).*R_fun(r,n,m);
end
mod_mode = mod(mode,2);
% Even polynomes
ind_m = find(mod_mode==0 & mv~=0);
for cpt=ind_m
n = nv(cpt);
m = mv(cpt);
fun(pupilLogical,cpt) = sqrt(n+1).*R_fun(r,n,m).*sqrt(2).*cos(m.*o);
end
% Odd polynomes
ind_m = find(mod_mode==1 & mv~=0);
for cpt=ind_m
n = nv(cpt);
m = mv(cpt);
fun(pupilLogical,cpt) = sqrt(n+1).*R_fun(r,n,m).*sqrt(2).*sin(m.*o);
end



end


function R = R_fun(r,n,m)
    R=zeros(size(r));
    for s=0:(n-m)/2
        R = R + (-1).^s.*prod(1:(n-s)).*r.^(n-2.*s)./...
            (prod(1:s).*prod(1:((n+m)/2-s)).*prod(1:((n-m)/2-s)));
    end


end