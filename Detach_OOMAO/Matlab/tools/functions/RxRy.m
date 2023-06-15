function [Rx, Ry] =  RxRy(fx,fy,fc,nActuator,D,r0,L0,fR0,modulation,binning,noiseVariance)
            nL      = nActuator - 1;
            d       = D/nL;
            f       = hypot(fx,fy);

            Wn      = noiseVariance/(2*fc)^2;
            Wphi    = spectrum(f,r0,L0,fR0);
            u       = fx;
           
            umod = 1/(2*d)/(nL/2)*modulation;
            Sx = zeros(size(u));
            idx = abs(u) > umod;
            Sx(idx) = 1i*sign(u(idx));
            idx = abs(u) <= umod;
            Sx(idx) = 2*1i/pi*asin(u(idx)/umod);
            Av = sinc(binning*d*u).*sinc(binning*d*u)';
            Sy = Sx.';

            %reconstruction filter
            AvRec = Av;
            SxAvRec = Sx.*AvRec;
            SyAvRec = Sy.*AvRec;
               
            % --------------------------------------
            %   MMSE filter
            % --------------------------------------
            gPSD = abs(Sx.*AvRec).^2 + abs(Sy.*AvRec).^2 + Wn./Wphi +1e-7;
            Rx = conj(SxAvRec)./gPSD;
            Ry = conj(SyAvRec)./gPSD;
        end