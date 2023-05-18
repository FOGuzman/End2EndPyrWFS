function out = sombrero(n,x)
            %% SOMBRERO Order n sombrero function
            %
            % out = sombrero(n,x) computes besselj(n,x)/x
            
            if n==0
                out = besselj(0,x)./x;
            else
                if n>1
                    out = zeros(size(x));
                else
                    out = 0.5*ones(size(x));
                end
                u = x~=0;
                x = x(u);
                out(u) = besselj(n,x)./x;
            end
        end