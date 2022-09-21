import torch
import torch.nn as nn

class RMSE(nn.Module):
    
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, inputs, targets):        
        tmp = (inputs-targets)**2
        loss =  torch.sqrt(torch.mean(tmp) )       
        return loss