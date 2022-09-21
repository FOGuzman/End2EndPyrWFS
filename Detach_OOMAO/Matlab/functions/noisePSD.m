function out = noisePSD(fx,fy,fc,Rx,Ry,noiseVariance,D)
            %% NOISEPSD Noise error power spectrum density
            out   = zeros(size(fx));
            if noiseVariance>0
                index = ~(abs(fx)>fc | abs(fy)>fc) & hypot(fx,fy)>0;
                %                f     = hypot(fx(index),fy(index));
                %                 out(index) = obj.noiseVariance./...
                %                     ( 2*pi*f.*tools.sinc(0.5*fx(index)/fc).*tools.sinc(0.5*fy(index)/fc)).^2;
                out(index) = noiseVariance/(2*fc)^2.*(abs(Rx(index)).^2 + abs(Ry(index)).^2);
                %out(index) = obj.noiseVariance*(abs(obj.Rx(index)).^2 + abs(obj.Ry(index).^2)); % See JM Conan thesis, Annex A
                out = out.*pistonFilter(D,hypot(fx,fy));
            end
                
end