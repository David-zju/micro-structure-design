from models.DDPM import DDPM
from models.ContextUnet import ContextUnet as Unet
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import pandas as pd

def str2list(str):
    num_str = str.split('[')[1].split(']')[0].split(',')
    num_list = [float(n) if '.' in n else int(n) for n in num_str] #int or float
    return num_list

class MinMaxScaler:
    def __init__(self):
        self.min_val = None
        self.max_val = None
        
    def fit(self, data: np.ndarray):
        self.min_val = np.min(data, axis=0)
        self.max_val = np.max(data, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        data_norm = (data - self.min_val) / (self.max_val - self.min_val)
        return data_norm

    def inverse_transform(self, data: torch.Tensor) -> np.ndarray:
        data_denorm = data * (self.max_val - self.min_val) + self.min_val
        return data_denorm.numpy()



class CustomDataset(Dataset):
    def __init__(self, data_path, scaler: MinMaxScaler) -> None:
        self.data = pd.read_csv(data_path)
        self.transform = None
        self.scaler = scaler
        
    def cal_transf(self):
        cols = ['E11', 'V12', 'G12']
        data = self.data[cols].values
        self.scaler.fit(data)
        data_norm = self.scaler.transform(data)
        self.data[cols] = data_norm
        return

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        row = self.data.iloc[index]
        exist = np.ones(8)
        x = np.array(str2list(row[13]), dtype=np.float32)
        thickness = np.ones(8)*row[10] # 现在都是厚度相同的情况，到时候需要修改
        
        if (x[-12:] == -1).all():
            exist[-4:] = 0
        if (x[0:12] == -1).all():
            exist[0:4] = 0
        
        # scale to [-1, 1]
        x[x==-1] = 5 # change default value
        x = (x-5)/5   
        x = x.reshape(8,3).T # transf to [3,8]
        thickness = ((thickness-0.75)*2)/1.1
        x = np.concatenate((x,thickness.reshape(1,8),exist.reshape(1,8)), axis=0, dtype=np.float32)
        if(x>-1).all() and (x<1).all(): # 似乎是没问题的
            breakpoint()
        condition1 = np.array(row[1:4], dtype=np.float32) # EvG
        condition2 = np.array(str2list(row[12]), dtype=np.float32) # Type
        conditions = np.concatenate((condition1, condition2), axis=0)
        #add new code for normalization of geometric information
        # row = self.data.iloc[index]
        # exist = np.ones(26)
        # x_list = str2list(row[13])
        # x = [ n/10 if n!=-1 else n for n in x_list] #the range of x is changed to {-1,0~1}
        # x = np.array( x, dtype=np.float32)
        # thickness = np.array([row[10],row[10]], dtype=np.float32) #should be modified later
        # thickness = ((thickness-0.2))/1.1 
        # if x[12]==x[13]==x[14] and x[12]==-1: 
        #     thickness[1] = -1
        #     #the range of thickness is changed to {-1,0~1}
        # x = np.concatenate((x, thickness), axis=0)
        # exist[x == -1] = 0
        # x = np.concatenate((x.reshape(1,26),exist.reshape(1,26)), axis=0, dtype=np.float32)
        # condition1 = np.array(row[1:4], dtype=np.float32) # EvG
        # condition2 = np.array(str2list(row[12]), dtype=np.float32) # Type
        # conditions = np.concatenate((condition1, condition2), axis=0)
        
        return x, conditions



def train(args:argparse.Namespace):
    device = try_device()

    data_path = 'data_process/all_data_augmented.csv'
    save_dir = 'train/4_29_aug/'
    scaler = MinMaxScaler()
    dataset = CustomDataset(data_path, scaler)
    dataset.cal_transf() # min_max transf of EGv
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    ddpm = DDPM.DDPM(nn_model = Unet.ContextUnet(in_channels=5, n_feat=args.n_feat, drop_prob=args.drop_prob),
                betas = (1e-4, 0.02), n_T = args.n_T, device = device, drop_prob = args.drop_prob)
    ddpm.to(device)
    
    lrate = args.l_rate
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    loss_save = []
    ddpm.train()
    for ep in range(args.n_epoch):
        print(f'epoch {ep}')
        loss_list = []
        
        optim.param_groups[0]["lr"] = lrate*(1-ep/args.n_epoch)
        pbar = tqdm(dataloader)
        loss_ema = None
        for x,c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else: 
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            loss_list.append(loss_ema)
            optim.step()
        loss_save.append(loss_list)
        if args.save_model and ep%10 == 0:
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")    
            df_loss = pd.DataFrame(loss_save)
            df_loss.to_csv(save_dir+'loss.csv', index=False)
    print("train finished")
    
    return 


def train_continue(model_dir:str, args:argparse.Namespace):
    device = try_device()

    data_path = 'data_process/all_data.csv'
    save_dir = 'train/4_25/'
    scaler = MinMaxScaler()
    dataset = CustomDataset(data_path, scaler)
    dataset.cal_transf() # min_max transf of EGv
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    ddpm = DDPM.DDPM(nn_model = Unet.ContextUnet(in_channels=1, n_feat=args.n_feat, drop_prob=args.drop_prob),
                betas = (1e-4, 0.02), n_T = args.n_T, device = device, drop_prob = args.drop_prob)
    ddpm.to(device)
    ddpm.load_state_dict(torch.load(model_dir))
    lrate = args.l_rate
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    loss_save = []
    ddpm.train()
    for ep in range(args.n_epoch):
        print(f'epoch {ep}')
        loss_list = []
        
        optim.param_groups[0]["lr"] = lrate*(1-ep/args.n_epoch)
        pbar = tqdm(dataloader)
        loss_ema = None
        for x,c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else: 
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            loss_list.append(loss_ema)
            optim.step()
        loss_save.append(loss_list)
        if args.save_model and ep%10 == 0:
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")  
            df_loss = pd.read_csv(save_dir+'loss.csv')
            df_append = pd.DataFrame(loss_save)
            df_loss = pd.concat([df_loss, df_append], axis=0)
            df_loss.to_csv(save_dir+'loss.csv', index=False)
            loss_save = []
    print("train finished")
    
    return



def try_device(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f"cuda:{i}")
    else:
        return torch.device("cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hyperparameter')
    parser.add_argument('--n_epoch', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=2560)
    parser.add_argument('--n_T', type=int, default=1000, help='扩散步数')
    parser.add_argument('--n_feat', type=int, default=512, help='number of feature in Unet')
    parser.add_argument('--l_rate', type=float, default=1e-3)
    parser.add_argument('--ws_test', type=list, default=[0.0, 0.5, 2.0, 4.0, 6.0], help='strength of generative guidance')
    parser.add_argument('--drop_prob', type=float, default=0.1)
    parser.add_argument('--save_model', type=bool, default=True)
    # print(type(parser.parse_args()))
    train(parser.parse_args())
    # train_continue("train/4_25/model_40.pth",parser.parse_args())
