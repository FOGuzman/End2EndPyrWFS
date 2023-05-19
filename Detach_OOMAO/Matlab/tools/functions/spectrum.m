function out = spectrum(f,r0,L0,fR0)
            %% SPECTRUM Phase power spectrum density
            %
            % out = phaseStats.spectrum(f,atm) computes the phase power
            % spectrum density from the spatial frequency f and an
            % atmosphere object
            %
            % See also atmosphere
            
            out = (24.*gamma(6./5)./5).^(5./6).*...
                (gamma(11./6).^2./(2.*pi.^(11./3))).*...
                r0.^(-5./3);
            out = out.*(f.^2 + 1./L0.^2).^(-11./6);
            out = sum([fR0]).*out;
        end