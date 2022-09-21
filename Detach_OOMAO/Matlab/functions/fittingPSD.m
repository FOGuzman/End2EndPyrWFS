function out = fittingPSD(fx,fy,fc,flagFitting,nTimes,r0,L0,fR0,D)
            %% FITTINGPSD Fitting error power spectrum density
            [fx,fy] = freqspace(size(fx,1)*nTimes,'meshgrid');
            fx = fx*fc*nTimes;
            fy = fy*fc*nTimes;
            out   = zeros(size(fx));
            if strcmp(flagFitting,'square')
                index  = abs(fx)>fc | abs(fy)>fc;
                
            else
                index  = hypot(fx,fy) > fc;
            end
            f     = hypot(fx(index),fy(index));
            out(index) = spectrum(f,r0,L0,fR0);
            out = out.*pistonFilter(D,hypot(fx,fy));
        end