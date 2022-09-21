import torch
import torch.nn as nn
import matplotlib.pyplot as plt  # This is python's popular plotting library.
# This is to ensure matplotlib plots inline and does not try to open a new window.

def showTensor(aTensor):
    plt.figure()
    plt.imshow(aTensor.numpy())
    plt.colorbar()
    plt.show()






class kNet(nn.Module):

    def __init__(self, args):
        super(kNet, self).__init__()
        self.Net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.Linear(16*(args.size[0])*(args.size[0]), 128),
            #nn.ReLU(),
            #nn.Dropout(0.5),
            #nn.Linear(128, 10),
            #nn.Linear(10, 1),
        )
        self.Linear = nn.Sequential(
            nn.Linear(16*(args.size[0])*(args.size[0]), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.Linear(10, 1),
            )
        



    def forward(self, meas_re, args):
        batch_size = meas_re.size(0)
        
        K = self.Net(meas_re)
        K = K.view(K.size(0), -1)
        K = self.Linear(K)
        
        #y = meas_re.view(meas_re.size(0), -1)
        nfeat = meas_re.size(1)*meas_re.size(2)*meas_re.size(3)
        y = torch.reshape(torch.transpose(meas_re,2,3),(batch_size,nfeat))
        y = torch.unsqueeze(y,-1)
        # = torch.unsqueeze(torch.transpose(y,0,1))
        z_coefs = torch.squeeze(torch.matmul(args.CM,y),-1)      
        #z_coefs = torch.transpose(z_coefs,1,0)
        zK = z_coefs*K

        return zK,z_coefs,K

