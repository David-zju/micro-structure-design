import torch
import torch.nn as nn
import numpy as np

class my_loss(nn.Module):
    def __init__(self) -> None:
        super(my_loss, self).__init__()
        self.loss_mse = nn.MSELoss()

    def forward(self, noise1, noise2, x):
        loss = self.loss_mse(noise1, noise2)
        # loss_coordinate = self.loss_mse(torch.clamp(x[:, :24], min=0, max=10), x[:, :24])
        loss_thickness = self.loss_mse(x[:, 24:], torch.zeros_like(x[:, 24:])) # we suppose the training data is correct
        # breakpoint()
        return loss + loss_thickness*0.4 # can be modified

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp() # 加法比乘法快
    
    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1) -> None:
        super(DDPM, self).__init__()
        # nn_model(x_i, conditions, t_is, void_mask)
        # condtions: [batch_size, *size], t_is: [batch_size, 1]
        self.nn_model = nn_model.to(device)
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k,v) #不用更新的参数值
        
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = my_loss()
    
    def forward(self, x, c):
        """
        used fot training, x is input vector, c are labels
        """
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)
        x_t = (
            self.sqrtab[_ts, None] * x
            + self.sqrtmab[_ts, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.
        
        # dropout context with some probability
        # context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        context_mask = torch.bernoulli(torch.zeros(c.shape[0])+self.drop_prob).to(self.device)
        # return MSE between added noise, and our predicted noise

        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask), x)
    
    def sample(self, size, device, conditions, guide_w = 0):
        n_sample = conditions.shape[0]
        x_i = torch.randn(n_sample, *size).to(device) # the vector to be processed
        context_mask = torch.zeros(conditions.shape[0]).to(device)
        void_mask = torch.ones(conditions.shape[0]).to(device)
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}', end='\r')
            t_is = torch.tensor([i/self.n_T]).to(device)
            eps_context = self.nn_model(x_i, conditions, t_is, context_mask)
            eps_void = self.nn_model(x_i, conditions, t_is, void_mask)
            eps = (1+guide_w)*eps_void - guide_w*eps_context
            z = torch.randn(n_sample, *size).to(device) if i>1 else 0
            
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
        
        return x_i

if __name__ == "__main__":
    x = list(range(24))+[0.2,0.2]
    x = torch.tensor([x,x])