import numpy
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Softmax

class C_Att(nn.Module):

    def __init__(self, in_dim, **kwargs):  #
        super().__init__()
        self.dim = in_dim
        self.Horizontal_Convd = nn.Conv2d(in_channels=in_dim,out_channels=2*in_dim,kernel_size=3,stride=2,padding=1)
        self.vertical_Convd = nn.Conv2d(in_channels=in_dim,out_channels=2*in_dim,kernel_size=3,stride=2,padding=1)
        self.Horizontal_Convu = nn.Conv2d(in_channels=2*in_dim,out_channels=in_dim,kernel_size=2,stride=2)
        self.vertical_Convu = nn.Conv2d(in_channels=2*in_dim,out_channels=in_dim,kernel_size=2,stride=2)
        self.Sakura_Mint = nn.Parameter(torch.zeros(1))
        self.x_dim = nn.Softmax(dim=1)
        self.H_dim = nn.Softmax(dim=2)
        self.V_dim = nn.Softmax(dim=3)



    def forward(self, x):
        # x is a list (Feature matrix, Laplacian (Adjcacency) Matrix).
        #assert isinstance(x, list)
        #x = torch.tensor(x)
        _,C_dim1,H_dim2,W_dim3 = x.shape
        #print(x.size())
        W_Vertical = x.permute(0,1,3,2).contiguous()
        W_Vertical = self.vertical_Convd(W_Vertical)
        W_Vertical = torch.relu(W_Vertical)
        W_Vertical = self.vertical_Convu(W_Vertical)
        Return_W = self.V_dim(W_Vertical)
        #a = a.reshape(W_dim3,H_dim2,C_dim1)
        H_horizontal = x.permute(0,3,1,2).contiguous()
        H_horizontal = self.Horizontal_Convd(H_horizontal)
        H_horizontal = torch.relu(H_horizontal)
        H_horizontal = self.Horizontal_Convu(H_horizontal)
        Return_H = self.H_dim(H_horizontal)
        x = self.x_dim(x.permute(0,3,2,1))
        #c = a.contiguous()
        #a = nn.Softmax(a)
        #b = c.permute(0,3,2,1)
        return self.Sakura_Mint*(Return_H+Return_W) + x

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    model = C_Att(3)
    x = torch.randn(1, 3, 3, 3)
    print(x)
    for i in range(2):
        out = model(x)
    print(out.shape)
    print(out)