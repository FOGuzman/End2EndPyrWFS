function out = piston(Npx,shape)
            
            x = -(Npx-1)/2:(Npx-1)/2;
            u = x;
            v = x;
            
            [x,y] = meshgrid(u,v);
            [o,r] = cart2pol(x,y);
           
            
            switch shape
                case 'disc'
                    out = double(r <= 1);
                case 'square'
                    out = double( abs(x)<=1 & abs(y)<=1 );
                case {'hex','hexagon'}
                    out = double( abs(x)<=sqrt(3)/2 & abs(y)<=x/sqrt(3)+1 & abs(y)<=-x/sqrt(3)+1 );
                otherwise
                    error('The piston shape is either a disc, a square or a hexagon')
            end
            
        end