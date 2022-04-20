import torch 
import torch.nn as nn
import numpy as np
import scipy.io as scio
from DisCorNet import *
import time 
from collections import OrderedDict

def zero_filling(x, factor = 16):
    H = x.size(0)
    W = x.size(1)
    D = x.size(2)

    newH = torch.ceil(torch.tensor(H / factor)) * factor
    newW = torch.ceil(torch.tensor(W / factor)) * factor
    newD = torch.ceil(torch.tensor(D / factor)) * factor

    tmp = torch.zeros(newH.int(), newW.int(),newD.int())

    pos = torch.zeros(2, 3)

    a = torch.ceil((newH - H) / 2)
    b = torch.ceil((newW - W) / 2)
    e = torch.ceil((newD - D) / 2)

    c = (a + H)
    d = (b + W)
    f = (e + D)

    a = a.int()
    b = b.int()
    c = c.int()
    d = d.int()
    e = e.int()
    f = f.int()

    tmp[a:c, b:d, e:f] = x

    pos[0, 0] = a
    pos[0, 1] = b
    pos[1, 0] = c
    pos[1, 1] = d
    pos[0, 2] = e
    pos[1, 2] = f

    return tmp, pos


def zero_removing(x, pos):

    a = pos[0, 0]
    b = pos[0, 1]
    c = pos[1, 0]
    d = pos[1, 1]
    e = pos[0, 2]
    f = pos[1, 2]

    a = a.int()
    b = b.int()
    c = c.int()
    d = d.int()
    e = e.int()
    f = f.int()

    x = x[a:c, b:d, e:f]
    return x


if __name__ == '__main__':

    with torch.no_grad():        
        print('Network Loading')
        ## load trained network 

        #Unet_chi = DisCorNet(7, (256, 256), ini_flag = False)

        #state_dict = torch.load('./DisCorNet_zero_init.pth', map_location=lambda storage, loc: storage)
        # create new OrderedDict that does not contain `module.`

        #new_state_dict = OrderedDict()
        #for k, v in state_dict.items():
         #   name = k[7:] # remove `module.`
         #   new_state_dict[name] = v
        # load params
        #Unet_chi.load_state_dict(new_state_dict)

        #Unet_chi.eval()

        Unet_chi = DisCorNet(7, (256, 256), ini_flag = False)
        Unet_chi = nn.DataParallel(Unet_chi)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Unet_chi.load_state_dict(torch.load('./DisCorNet_zero_init.pth'))
        Unet_chi.to(device)
        Unet_chi.eval()


        for idx in range(0, 300):
            mask_name = "./testing_simu_brain_all/test" + str(idx+1) + '.mat'
            matImage = scio.loadmat(mask_name)
            ksp = matImage['ksp']
            ksp = np.array(ksp)

            ksp = torch.from_numpy(ksp)

            print(ksp.dtype)

            imsize = ksp.shape

            #ksp = ksp.complex()
            
            ksp = torch.unsqueeze(ksp, 0)

            ksp = torch.unsqueeze(ksp, 0)

            ksp = ksp.to(device)

            traj = matImage['traj']
            traj = np.array(traj)
            traj = torch.from_numpy(traj).float()
            traj = traj.to(device)

            ind = matImage['ind']
            ind = np.array(ind)
            ind = torch.from_numpy(ind).float()
            ind = ind.to(device)

            AH_op = nufft.KbNufft(im_size = imsize, device = device) ## transposed operator: AH; 
            Ahb = AH_op(ksp, traj)/(256*256)
            Ahb = torch.reshape(Ahb, [1, 1, imsize[0], imsize[1]])

            Ahb2 = Ahb.to('cpu')
            Ahb2 = Ahb2.numpy()
            
            ksp = torch.reshape(ksp, [1, 1, imsize[0], imsize[1]])
            ksp = ksp.to('cpu')
            ksp = ksp.numpy()



            print('Saving results')

            path = "./initial_" +  str(idx+1) + '.mat'
            scio.savemat(path, {'Ahb2':Ahb2})
            print('end')

            Ahb_r = np.real(Ahb2)
            Ahb_r = torch.from_numpy(Ahb_r).float() 
            Ahb_r = torch.squeeze(Ahb_r, 0)

            Ahb_i = np.imag(Ahb2)
            Ahb_i = torch.from_numpy(Ahb_i).float()
            Ahb_i = torch.squeeze(Ahb_i, 0)

            Ahb = torch.cat([Ahb_r, Ahb_i], dim = 0).unsqueeze(0)

            Ahb = Ahb.to(device)

            ksp_r = np.real(ksp)
            ksp_r = torch.from_numpy(ksp_r).float() 
            ksp_r = torch.squeeze(ksp_r, 0)

            ksp_i = np.imag(ksp)
            ksp_i = torch.from_numpy(ksp_i).float()
            ksp_i = torch.squeeze(ksp_i, 0)


            ksp = torch.cat([ksp_r, ksp_i], dim = 0).unsqueeze(0)
            ksp = ksp.to(device)
            print(Ahb.shape)

            smaps = torch.ones(1, 2, 256, 256).float()
            R_cal_OP = R_cal(imsize, traj, Ahb.cuda(), ind, ksp, device)
            R_cal_OP = nn.DataParallel(R_cal_OP)


            print('reconing ...')
            
            time_start = time.time()

            pred_chi, _, _ = Unet_chi(Ahb.cuda(), R_cal_OP, device)

            pred_chi = R2C(pred_chi) 
            #pred_chi = torch.real(pred_chi)
            #print(pred_chi.shape)
            
            pred_chi = torch.squeeze(pred_chi, 0)
            pred_chi = torch.squeeze(pred_chi, 0)

            time_end =time.time()
            print(time_end - time_start)

            pred_chi = pred_chi.to('cpu')
            pred_chi = pred_chi.numpy()

            print('Saving results')

            path = "./DisCorNet_zf_" +  str(idx+1) + '.mat'
            print(path)
            scio.savemat(path, {'pred_chi':pred_chi})
            print('end')