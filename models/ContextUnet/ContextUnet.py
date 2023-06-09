import torch
import torch.nn as nn
import numpy as np

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
        super(ResidualConvBlock, self).__init__()
        # standard ResNet style convolutional block
        self.activate_func = nn.GELU()
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm1d(out_channels),
            self.activate_func,
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm1d(out_channels),
            self.activate_func,
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels:int, out_channels:int) -> None:
        super(UnetDown, self).__init__()
        layers = [ResidualConvBlock( in_channels, out_channels), nn.Conv1d(out_channels, out_channels, 2,2,0)]
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels:int, out_channels:int) -> None:
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose1d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
            nn.GELU()
            # nn.ReLU()
        ]
        self.model = nn.Sequential(*layers)
    def forward(self, x, shortcut:torch.Tensor = None):
        if(shortcut != None):
            x = torch.cat((x, shortcut), 1)    
        return self.model(x)


class oldUnetDown(nn.Module):
    def __init__(self, in_channels:int, out_channels:int) -> None:
        super(UnetDown, self).__init__()
        layers = [ResidualConvBlock( in_channels, out_channels), nn.MaxPool1d(2) ]
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    
class oldUnetUp(nn.Module):
    def __init__(self, in_channels:int, out_channels:int) -> None:
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose1d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
            nn.GELU()
            # nn.ReLU()
        ]
        self.model = nn.Sequential(*layers)
    def forward(self, x, shortcut:torch.Tensor = None):
        if(shortcut != None):
            x = torch.cat((x, shortcut), 1)    
        return self.model(x)
        
class Embed(nn.Module):
    def __init__(self, input_dim, emb_dim) -> None:
        super(Embed, self).__init__()
        self.input_dim = input_dim
        self.activate_function = nn.GELU()
        layers = [
            nn.Linear(input_dim, emb_dim),
            self.activate_function,
            nn.Linear(emb_dim, emb_dim)
        ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x).unsqueeze(2)
        

class ContextUnet(nn.Module):
    def __init__(self, in_channels, drop_prob, n_feat = 8):
        super(ContextUnet, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.drop_prob = drop_prob
        self.cd_dim = 9
        self.init_conv = ResidualConvBlock(in_channels, n_feat)
        self.down1 = UnetDown(in_channels=n_feat, out_channels=n_feat*2)
        self.down2 = UnetDown(in_channels=n_feat*2, out_channels=n_feat*4)
        self.down3 = UnetDown(in_channels=n_feat*4, out_channels=n_feat*8)
        self.up0 = UnetUp(in_channels=n_feat*8, out_channels=n_feat*4)
        self.up1 = UnetUp(in_channels=n_feat*8, out_channels=n_feat*2)
        self.up2 = UnetUp(in_channels=n_feat*4, out_channels=n_feat)
        self.up3 = nn.Sequential(
            nn.Conv1d(2*n_feat, n_feat, 3, 1,1),
            nn.GroupNorm(4, n_feat),
            nn.GELU(),
            nn.Conv1d(n_feat, 5, 3, 1, 1),
        )
        
        self.phy_emb1 = Embed(self.cd_dim, n_feat*4)
        self.phy_emb2 = Embed(self.cd_dim, n_feat*2)
        self.t_emb1 = Embed(1, n_feat*4)
        self.t_emb2 = Embed(1, n_feat*2)
        self.x_emb_in = nn.Sequential(nn.Linear(8, 8),
                                      nn.BatchNorm1d(5),
                                   nn.GELU(),
                                   nn.Linear(8, 8))
        self.x_emb_out = nn.Sequential(nn.Linear(8, 8),
                                #    nn.ReLU(),
                                   nn.BatchNorm1d(5),
                                   nn.GELU(),
                                   nn.Linear(8, 8))

    def forward(self, x:torch.Tensor, conditions:torch.Tensor, t:torch.Tensor, context_mask:torch.Tensor):
        conditions = conditions * context_mask.unsqueeze(1)
        t = t.unsqueeze(1)
        # x = x.unsqueeze(1)
        # embed
        x = self.x_emb_in(x)
        cemb1 = self.phy_emb1(conditions)
        cemb2 = self.phy_emb2(conditions)
        temb1 = self.t_emb1(t)
        temb2 = self.t_emb2(t)

        x = self.init_conv(x) # n_feat
        d1 = self.down1(x) # n_feat*2
        d2 = self.down2(d1) # n_feat*4
        d3 = self.down3(d2) # n_feat*8
        
        up1 = self.up0(d3) # n_feat*4
        up2 = self.up1(up1*cemb1+temb1, d2) # n_feat*2
        up3 = self.up2(up2*cemb2+temb2, d1) # n_feat
        up4 = self.up3(torch.cat((up3, x), 1)) # 5
        
        out = self.x_emb_out(up4)
        # breakpoint()
        return out
    
if __name__ == "__main__":
    batch_size = 2
    x = torch.rand([batch_size, 5, 8])
    _ts = torch.randint(1,11,(x.shape[0],))
    conditions = torch.rand([batch_size,9])
    context_mask = torch.bernoulli(torch.zeros(conditions.shape[0])+0.5)
    trans = ContextUnet(in_channels=5, n_feat=128, drop_prob=0.5)
    trans(x, conditions, _ts/10, context_mask)
    
    
    # transformer = nn.Transformer(d_model = 1, nhead=1, num_encoder_layers=12, num_decoder_layers=12, activation=torch.tanh, dropout=0.1)
    # src = torch.rand((20, 32)).unsqueeze(2)
    # tgt = torch.rand((10, 32)).unsqueeze(2)
    # # src = torch.rand((20, 32, 256))
    # # tgt = torch.rand((10, 32, 256))
    # out = transformer(src, tgt)
    # print(out.size())