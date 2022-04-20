"""
******************************************************************
# DisCornet: 
# Unrolling ISTA algorithm for MRI distortion correction problems. 
******************************************************************
"""

## *******import packages********
from operator import index
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchkbnufft as nufft 

import scipy.io as sio
import numpy as np
import os
import time
## *************end*************

# define DisCorNet achitecture
# Define ISTA-Net-plus
class DisCorNet(torch.nn.Module):
    def __init__(self, LayerNo, imsize, ini_flag = False, num_in = 16, num_out = 16, ks = 3, pad =1):
        super(DisCorNet, self).__init__()
        """
        input parameters: 
        LayerNo: number of iteration times (layers); 
        imsize: image size; 
        ini_flag: use 0 or AH*y as the initial guess; 
        """

        self.imsize = imsize
        self.ini_flag = ini_flag

        layers = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            layers.append(OneIter(imsize, num_in, num_out, ks = 3, pad =1))

        self.nets = nn.ModuleList(layers)

    def forward(self, AHy, R_cal, device = 'cuda'):
        """
        input parameters: 
        y: acquired kspace data in size of Nb * 2 * Nx * Ny, where the first channel is the real component, 
            while the second channel is the imaginary part; 
        ktraj: kspace trajectory; 
        """
        x = torch.zeros(AHy.shape)
        x = x.to(device)


        if self.ini_flag:
            x = AHy
        
        #print(AHy.shape)
        AHy = R2C(AHy)
        #print(AHy.shape)

        sym_loss_all = []   # for computing symmetric loss
        DF_loss_all = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            x, layer_sym, layer_DF = self.nets[i](x, R_cal)
            sym_loss_all.append(layer_sym)
            DF_loss_all.append(layer_DF)

        x_final = x

        return x_final, sym_loss_all, DF_loss_all


# define one iteration Layers
class OneIter(torch.nn.Module):
    def __init__(self, imsize,  num_in = 16, num_out = 16, ks = 3, pad =1):
        super(OneIter, self).__init__()

        self.alpha = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
        self.length = imsize[0] * imsize[1]

        self.conv_D = nn.Conv2d(2, num_in, ks, padding= pad)

        self.H_for = H_Operator(num_in = num_in, num_out = num_out, ks = 3, pad = 1)

        self.H_back = H_Operator(num_in = num_out, num_out = num_out, ks = 3, pad = 1)

        self.conv_G = nn.Conv2d(num_out, 2, ks, padding= pad)

    def SoftThr(self, x):
        ## soft thresholding; 
        x = torch.mul(torch.sign(x), F.relu(torch.abs(x) - self.soft_thr)) 

        return x

    def forward(self, x, R_cal):

        x_input, DF_loss, x_updated = R_cal(x, self.alpha)
        
        x = x_updated

        x_D = self.conv_D(x)

        x_forward = self.H_for(x_D)

        x = self.SoftThr(x_forward)

        x_backward = self.H_back(x)

        x_G = self.conv_G(x_backward)

        x_pred = x_input + x_G

        x_D_est = self.H_back(x_forward)
        symloss = x_D_est - x_D

        return x_pred, symloss, DF_loss

# define H_forward/H_backward_operator; 
class H_Operator(torch.nn.Module):
    def __init__(self, num_in = 16, num_out = 16, ks = 3, pad = 1):
        super(H_Operator, self).__init__()

        """
        use two consecutive convolutional layers, to learn the optimal non-linear forward and backward transformation; 
        An additional loss function should be added in the training process to force the H_forwad * H_backward = I; 
        """
        self.conv1 = nn.Conv2d(num_in, num_out, ks, padding= pad)
        ##self.bn = nn.BatchNorm2d(num_out)  ## batch-normalization is not necessary; 
        self.relu = nn.ReLU(inplace = True)

        self.conv2 = nn.Conv2d(num_out, num_out, ks, padding= pad)

    def forward(self, x):

        x = self.conv1(x)
        ##x = self.bn(x) ## batch-normalization is not necessary; 
        x = self.relu(x)

        x = self.conv2(x)

        return x

## Physical-modle Block (PMB): where the data fidelity model were involved; 
class R_cal(torch.nn.Module):
    def __init__(self, imsize, ktraj, Ahy, ind, ksp, device):
        super(R_cal, self).__init__()
        """
        imsize: image size; 
        """
        self.imsize = imsize
        self.length = imsize[0] * imsize[1]

        self.Ahy = Ahy.to(device)
        self.Ahy = R2C(self.Ahy)
	    
        self.ksp = ksp.to(device) 
        self.ksp = R2C(self.ksp)

        xx = torch.zeros(1, 1, 256, 256).float()
        self.xx = xx.to(device)


        ## need to adjuest the network structure to make it feasible for training; 
        ## USE ToepNufft for furthure acceleration; 
        self.ktraj = ktraj
        self.ind = ind
     
        self.AAH_op = nufft.ToepNufft() ## AH * A operator. 
        self.kernel = nufft.calc_toeplitz_kernel(ktraj, im_size = imsize)
        self.AHA_op = self.AAH_op.to(device)
        self.kernel = self.kernel.to(device)
        
        self.AH_op = nufft.KbNufftAdjoint(im_size = imsize) ##  transposed operator: AH;
        self.AH_op = self.AH_op.to(device)
        self.A_op = nufft.KbNufft(im_size = imsize) ## forward operator: matrix A; 
        self.A_op = self.A_op.to(device) 

    def forward(self, x, alpha):
        """
        R_k = x_k - alpha * AH * A * x_k + alpha * AH *y)
        R_k = x_k - alpha * (AH * A * x_k - AH *y)
        """
        x = R2C(x)  ## convert real data into complex data;
        #INPUT = x  ## x_k
        #smaps = R2C(smaps)
        nn = x.shape
        nb = nn[0]

        x = torch.reshape(x, [nb, 1, 256 * 256])
        x = self.AH_op(x, self.ktraj)
        x = torch.reshape(x, [nb, 1, 256, 256])
        mask3 = (self.ind-1)*(-1)
        mask3 = mask3.type(torch.bool)
        mask4 = mask3[0, :]
        x[:, :, mask4,:] = self.ksp[:, :, mask4,:]
        x = self.A_op(x, self.ktraj)/(self.length)
        x = torch.reshape(x, [nb, 1, 256, 256])   

        x =  torch.abs(x)+1j*self.xx
       
        INPUT = x
        
        INPUT2 = C2R(x)

        #xxx = self.AAH_op(x, self.kernel, smaps) / (self.length)  ## Ah * A * x_k
        x = torch.reshape(x, [nb, 1, 256 * 256])
        #self.ktraj = ktraj
        x = self.AH_op(x, self.ktraj)
        x = torch.reshape(x, [nb, 1, 256, 256])
        mask = self.ind
        #print(mask.shape)
        mask = mask.type(torch.bool)
        mask2 = mask[0, :]
        #print(mask2)
        #print(mask2.shape)
        x[:, :, mask2,:] = 0

        x = self.A_op(x, self.ktraj)/(self.length)
        x = torch.reshape(x, [nb, 1, 256, 256])

        x =  torch.abs(x)+1j*self.xx

        DF_loss = x - torch.abs(self.Ahy)  ## Ah * A * x_k - Ah * y

        R = INPUT - alpha * DF_loss
        
        R = C2R(R) ## convert complex data back into real data;
        DF_loss = C2R(DF_loss)
        return R, DF_loss, INPUT2

################ basic utility functions ###################
def R2C(x):
    """
    input dat: 
    x, real tensor in the size of Nb * 2 * Nx * Ny, with the real component as the first channel data, 
    while imaginary data as the second channel data;
    output data:
    x, complex data in the size of Nb * 1 * Nx * Ny; 
    """
    x = x.permute(0, 2, 3, 1).contiguous() ## Nb * Nx * Ny * 2, real; 
    x = torch.view_as_complex(x) # Nb * Nx * Ny, complex; 
    x = torch.unsqueeze(x, dim = 1) # Nb * 1 * Nx * Ny, complex; 
    return x 

def C2R(x):
    """
    input data: 
    x, complex tensor in the size of Nb * 1 * Nx * Ny
    output data: 
    x, real tensor in the size of Nb * 2 * Nx * Ny
    """
    x = torch.squeeze(x, dim = 1) # Nb * Nx * Ny, complex; 
    x = torch.view_as_real(x).contiguous()  ## Nb * Nx * Ny * 2, real; 
    x = x.permute(0, 3, 1, 2)  ## Nb * 2 * Nx * Ny, real; 
    return x

############# non-critical utility functions ###################
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(m.bias)   
    if isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)   

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

#################### For Code Test ##################################
## before running the training codes, verify the network architecture. 
if __name__ == '__main__':
   
   ## make this version works well with batch-ed data. 
    AHy = torch.randn(2, 2, 100, 100).float()

    smaps = torch.ones(AHy.shape) 
    
    traj = torch.randn(2, 2, 100 * 100).float()

    dcnet = DisCorNet(5, (100, 100))
    dcnet.apply(weights_init)

 
    #print(dcnet.state_dict)
    print(get_parameter_number(dcnet))

    torch.cuda.current_device()
    torch.cuda._initialized = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dcrnet = dcnet.to(device)
    R_cal_OP =  R_cal((100, 100), traj,AHy, device)
    smaps = smaps.to(device)

    AHy = AHy.to(device)
    traj = traj.to(device)
    print('input' + str(AHy.size()))
    #x_final, floss = dcnet(AHy2, traj2, R_cal_OP)
    a = time.time()
    x_final, floss, D_loss = dcnet(AHy, R_cal_OP, smaps)
    b = time.time()
    print(b - a)
    print('output'+str(x_final.size()))