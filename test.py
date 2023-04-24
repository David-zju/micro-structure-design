import torch

# 假设有四个长度为10的一维向量组成batch
batch = [
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
]

# 将batch中的一维向量拼接成二维张量
input_tensor = torch.tensor(batch)

# 在第一维添加batch size
input_tensor = input_tensor.unsqueeze(0)

# 创建Transformer模型
model = torch.nn.Transformer(d_model=10, nhead=2)

# 将输入传入模型
output_tensor = model(input_tensor, input_tensor)
