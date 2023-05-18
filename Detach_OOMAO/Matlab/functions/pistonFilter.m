function out = pistonFilter(D,f)
            red = pi*D*f;
            out = 1 - 4*sombrero(1,red).^2;
            
end