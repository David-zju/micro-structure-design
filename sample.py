from models.DDPM import DDPM
from models.ContextUnet import ContextUnet as Unet
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import pandas as pd
import os
from train import *

def output_transf(x_gen:torch.Tensor)-> torch.Tensor:
    scaler = MinMaxScaler()
    # x_np = x.to('cpu').numpy()
    # x = x.to('cpu')
    batch_size = x_gen.shape[0]
    exist_x = torch.round(x_gen[:,4,:]).repeat_interleave(3, dim=1) # [batch_size, 24]
    exist_t = torch.round(x_gen[:,4,:].reshape(-1,2,4).mean(dim=2)) # [batch_size, 2]
    
    x = x_gen[:,0:3,:]
    x = torch.transpose(x,1,2).reshape(-1,24) #[batch_size, 24]
    x = (torch.clamp(x, -1, 1) + 1) * 5
    x[exist_x == 0] = -1
    
    thickness = x_gen[:,3,:]
    thickness1 = torch.mean(thickness[:,:4],dim=1)
    thickness2 = torch.mean(thickness[:,-4:], dim=1)
    thickness = torch.concat((thickness1.reshape(batch_size,1), thickness2.reshape(batch_size,1)),dim=1) # [batch_size, 2]
    thickness = (torch.clamp(thickness, -1, 1)/2)*1.1 + 0.75 
    thickness[exist_t == 0] = -1
    # breakpoint()
    return torch.concat((x, thickness), dim=1)
    
def sample(model_path, output_path, conditions=[]):
    device = try_device()

    test_df = pd.read_csv(output_path)
    for row in test_df.itertuples(index=True):
        conditions.append([row.E11, row.V12, row.G12]+str2list(row.Type))

    data_path = 'data_process/all_data_augmented.csv'
    scaler = MinMaxScaler()
    dataset = CustomDataset(data_path, scaler)
    dataset.cal_transf() # min_max transf of EGv
    conditions = np.array(conditions, dtype=np.float32)
    conditions[:,0:3] = scaler.transform(conditions[:,0:3])

    ddpm = DDPM.DDPM(nn_model = Unet.ContextUnet(in_channels=5, n_feat=512, drop_prob=0.1),
                betas = (1e-4, 0.02), n_T = 1000, device = device, drop_prob = 0.1)
    ddpm.to(device)
    ddpm.load_state_dict(torch.load(model_path))

    ws_test = np.arange(0.0,2.5,0.5).tolist()
    ddpm.eval()
    with torch.no_grad():
        for w in ws_test:
            print(f"w = {w}")
            x_gen = ddpm.sample([5,8], device, torch.tensor(conditions).to(device), guide_w=w)
            x_gen = output_transf(x_gen)
            test_df['Geo_'+str(w)] = x_gen.tolist()
            breakpoint()

    test_df.to_csv(output_path, index=False)
    return test_df

def Convert_format(test_df:pd.DataFrame):
    folder_dir = 'generate/'
    for row in test_df.itertuples(index=True):
        geo_vec = eval(row[10])
        UcName = "E" + str(row.E11) + 'v' + str(row.V12) + "G" + str(row.G12) + '.txt'
        save_dir = os.path.join(folder_dir, UcName)
        f = open(save_dir, 'w+')
        f.write("{0:<10}{1:<10}\n".format('Thickness1', 'Thickness2'))
        f.write('{0:<10.5f}{1:<10.5f}\n'.format(geo_vec[24], geo_vec[25]))
        f.write("{0:<10}{1:<10}{2:<10}{3:<10}\n".format('FACE0', 'X', 'Y', 'Z'))
        for i in range(0,4):
            f.write('{0:<10}{1:<10.5f}{2:<10.5f}{3:<10.5f}\n'.format('POINT', geo_vec[0+3*i], geo_vec[1+3*i], geo_vec[2+3*i]))
        # f.write("FACE1    X         Y         Z         \n")
        f.write("{0:<10}{1:<10}{2:<10}{3:<10}\n".format('FACE1', 'X', 'Y', 'Z'))
        for i in range(4,8):
            f.write('{0:<10}{1:<10.5f}{2:<10.5f}{3:<10.5f}\n'.format('POINT', geo_vec[0+3*i], geo_vec[1+3*i], geo_vec[2+3*i]))
        f.close()
    print("convert finished")
    

if __name__ == "__main__":
    model_dir = 'train/4_29_aug/model_1350.pth'
    output_dir = 'generate\output.csv'
    # conditions.shape = [n_samples, features=(9, 1)]
    # test_df = sample(model_dir, output_dir)
    test_df = pd.read_csv(output_dir)
    Convert_format(test_df)