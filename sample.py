from models.DDPM import DDPM
from models.ContextUnet import ContextUnet as Unet
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import pandas as pd
from train import *

def output_transf(x:torch.Tensor)-> torch.Tensor:
    scaler = MinMaxScaler()
    x_np = x.to('cpu').numpy()
    # scaler.fit(x_np)
    # x_norm = torch.tensor(scaler.transform(x_np))
    # scale = torch.cat([10*torch.ones(1,24), torch.ones(1,2)], dim=1)
    x_np = x_np*20
    x_np[x_np<10] = 9
    x_np = x_np-10
    x_np[x_np > 10] = 10
    return torch.Tensor(x_np)
    



def sample(model_path, output_path, conditions=[]):
    device = try_device()

    test_df = pd.read_csv(output_path)
    for row in test_df.itertuples(index=True):
        conditions.append([row.E11, row.V12, row.G12]+str2list(row.Type))

    data_path = 'data_process/all_data.csv'
    scaler = MinMaxScaler()
    dataset = CustomDataset(data_path, scaler)
    dataset.cal_transf() # min_max transf of EGv
    conditions = np.array(conditions, dtype=np.float32)
    conditions[:,0:3] = scaler.transform(conditions[:,0:3])

    ddpm = DDPM.DDPM(nn_model = Unet.ContextUnet(in_channels=1, n_feat=256, drop_prob=0.1),
                betas = (1e-4, 0.02), n_T = 1000, device = device, drop_prob = 0.1)
    ddpm.to(device)
    ddpm.load_state_dict(torch.load(model_path))

    ws_test = np.arange(0.0,2.5,0.5).tolist()
    ddpm.eval()
    with torch.no_grad():
        for w in ws_test:
            print(f"w = {w}")
            x_gen = ddpm.sample([26], device, torch.tensor(conditions).to(device), guide_w=w)
            x_gen = output_transf(x_gen)
            test_df['Geo_'+str(w)] = x_gen.tolist()

    test_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    model_dir = 'train/4_25/model_930.pth'
    output_dir = 'generate\output.csv'
    # conditions.shape = [n_samples, features=(9, 1)]
    sample(model_dir, output_dir)