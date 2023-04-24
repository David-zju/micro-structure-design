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
