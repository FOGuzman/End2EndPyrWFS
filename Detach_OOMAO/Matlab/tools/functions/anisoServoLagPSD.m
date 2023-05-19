function [out, rtf] = anisoServoLagPSD(fx,fy,fc,Rx,Ry,SxAv,SyAv,r0,L0,fR0,D)
            %% SERVOLAGPSD Servo-lag power spectrum density

            iSrc=1;
            out   = zeros(size(fx));
            index = ~(abs(fx)>=fc | abs(fy)>=fc);
            pf = pistonFilter(D,hypot(fx,fy));
            fx     = fx(index);
            fy     = fy(index);
%             if ~isempty(obj.src)
%                 zLayer = [obj.atm.layer.altitude];
%                 fr0    = [obj.atm.layer.fractionnalR0];
%                 A = zeros(size(fx));
%                 for kLayer=1:obj.atm.nLayer
%                     red = 2*pi*zLayer(kLayer)*...
%                         ( fx*obj.src(iSrc).directionVector(1) + fy*obj.src(iSrc).directionVector(2) );
%                     A  = A + fr0(kLayer)*exp(1i*red);
%                 end
%             else
               % A = ones(size(fx));
%             end
            %out(index) = phaseStats.spectrum(hypot(fx,fy),obj.atm).*averageClosedLoopRejection(obj,fx,fy);
           % F = (Rx(index).*SxAv(index) + Ry(index).*SyAv(index));
            rtf = 1;
            out(index) =  rtf.*spectrum(hypot(fx,fy),r0,L0,fR0);
            out = pf.*real(out);
        end