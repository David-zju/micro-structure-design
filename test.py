import torch

x = torch.randn(2, 8)

# reshape the tensor into (batch_size, 2, 4)
x = x.reshape(-1, 2, 4)

# compute the mean in dim=2
mean1 = x.mean(dim=2)
mean2 = x.mean(dim=2)
breakpoint()
# concatenate the means along dim=1
means = torch.cat([mean1, mean2], dim=1)

print(means.shape)  # (32, 2)
