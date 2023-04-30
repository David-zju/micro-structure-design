epoch10000：
```python
parser = argparse.ArgumentParser(description='hyperparameter')
parser.add_argument('--n_epoch', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=4000)
parser.add_argument('--n_T', type=int, default=500, help='扩散步数')
parser.add_argument('--n_feat', type=int, default=8, help='number of feature in Unet')
parser.add_argument('--l_rate', type=float, default=1e-4)
parser.add_argument('--ws_test', type=list, default=[0.0, 0.5, 2.0], help='strength of generative guidance')
parser.add_argument('--drop_prob', type=float, default=0.01)
parser.add_argument('--save_model', type=bool, default=True)
```
add_non_negative_loss:
主要是把lrate调高了一下，之前收敛的有点慢了
```python
    parser = argparse.ArgumentParser(description='hyperparameter')
    parser.add_argument('--n_epoch', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--n_T', type=int, default=500, help='扩散步数')
    parser.add_argument('--n_feat', type=int, default=8, help='number of feature in Unet')
    parser.add_argument('--l_rate', type=float, default=1e-3)
    parser.add_argument('--ws_test', type=list, default=[0.0, 0.5, 2.0], help='strength of generative guidance')
    parser.add_argument('--drop_prob', type=float, default=0.01)
    parser.add_argument('--save_model', type=bool, default=True)
    thickness: 0.2
```

4_25:
增加了feature和扩散步数（然后增加了输入的归一化）
```python
    parser = argparse.ArgumentParser(description='hyperparameter')
    parser.add_argument('--n_epoch', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--n_T', type=int, default=1000, help='扩散步数')
    parser.add_argument('--n_feat', type=int, default=256, help='number of feature in Unet')
    parser.add_argument('--l_rate', type=float, default=5e-4)
    parser.add_argument('--ws_test', type=list, default=[0.0, 0.5, 2.0], help='strength of generative guidance')
    parser.add_argument('--drop_prob', type=float, default=0.1)
    parser.add_argument('--save_model', type=bool, default=True)
```
add_mask: 和上面一样

4_27 用的是一样的训练参数，然后现在目前收敛到0.05左右（2000epoch情况下）

4_29
因为模型和训练数据的增加，每个epoch迭代次数其实很多，把epoch数降到3000
相应的batchsize和feature都进行了一定的调整
```python
    parser = argparse.ArgumentParser(description='hyperparameter')
    parser.add_argument('--n_epoch', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=2560)
    parser.add_argument('--n_T', type=int, default=1000, help='扩散步数')
    parser.add_argument('--n_feat', type=int, default=512, help='number of feature in Unet')
    parser.add_argument('--l_rate', type=float, default=5e-4)
    parser.add_argument('--ws_test', type=list, default=[0.0, 0.5, 2.0, 4.0, 6.0], help='strength of generative guidance')
    parser.add_argument('--drop_prob', type=float, default=0.1)
    parser.add_argument('--save_model', type=bool, default=True)
```