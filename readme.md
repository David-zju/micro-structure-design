input vector:
geometry
|f1|f2|t1|t2|
|--|--|--|--|
|(p1,p2,p3,p4)|(p1,p2,p3,p4)|t1|t2|

condition
|E|G|v|Type|
|--|--|--|--|
|E|G|v|onehot{i1,i2,i3,i4,i5,i6}|


### 2023/4/22
- [x] 几何信息的更新 
- [x] 输出了第一版数据，但是出现了很多负数，感觉训练数据中-1这一项会对神经网络产生影响？  
    如何对-1进行处理(先置0)
- [x] 数值范围需要限定，例如厚度不能太厚，以及$p_i$的值在0-10范围内  
  现在是在损失函数内加了1项，其中权重到时候需要再调整调整，目前如果将-1改成0之后训练数据都落在0-10这个范围内了，所以直接做损失函数也没有意义
- [ ] 
### 2023/4/24
今天发现还是有负数，开始溯源
通过激活函数的修改使其能够不出现负值
其次，添加第一层的unet连接（up4与x），并更新了中间几层的feature的值
在输出层添加了GroupNorm层（但愿能有用）
---
晚上检查后发现还是没有用的，思考，是否应该输出直接加一个Relu层，但是这是无济于事的，因为实际上是采样的过程生成样本，而不是网络直接去学（这个不是gan）
先在图像上做做试验，发现如果采样时间步长如果很短，那么输出的图片就会像是散点图（10 vs 500），那么怎么样才能评估超参数呢？

跟师姐讨论了一下，模型确实可能会有问题，但是先试试超参数上的调整
最后的方案是使用模型的种类来训练，不使用扩散模型，直接插值

查看图像生成的代码，torchvision里面有个make_grid函数，会对输出的tensor归一化，然后*255，那么我们也可以对我们的输出数据进行归一化，然后*一个我们自己的scaler，先试试看

### 2023/4/25
早上与师姐讨论，师姐有个想法是通过神经网络输出的值域来控制逆采样过程的$x_t$的值域，可能需要验证一下数学公式上是否满足这个性质
区分坐标的0和没有面片的0，坐标+10->[10,20]
坐标的输入也归一化一下
今天看了一下，make_grid函数实际上没有启用归一化，是save_image的时候直接截断了（我们是否可以改成截断？）
把板厚和坐标拆开？

输入的定义域是[-1,1] 我们把缺省值定为-1，然后使用[-0.9, 1]作为可行的区域，输出的时候把小于-0.9的全截断为-1（这样或许可以减少出错的概率）
另外一个方向是，把输出的坐标挪到关键点上，关键点的位置事先给定，相当于再加一个分类器变成离散的结果，可以保证结果的正确性

这样输入的定义仍然不太好，采用分开通道的模式，把是否缺省放在另外一个通道，缺省为0，不缺省为1，然后把所有的输入映射到[-1,1]
sample函数仍然需要改一下，明天再说，先下班
（看了一下，现在的loss又下降了一个台阶，不知道是不是输出参数变多还是模型更好了）

### 2023/4/26
早上与师姐讨论，发现Glide里面的text embed也用了mask的思想
sample函数内，mask需要取整，使用round函数（以0.5为分界线），输出坐标的时候，先clamp到[-1,1]，再放缩到原来的[0,10]，厚度也是一样

板的mask，sample输出缺省应该为-1而不是0(这个改完了)
现在模型输出的几个sample学习到了数据的大体分布特征，（例如在几个5，10，2.5之类的值附近），为了满足接口的要求，我们可以把表面上的点挪动到关键点上
关键面片由四个点组成，但是实际上生成的数据是不共面的，我们得开发一个基于4个点生成一个最接近的面的算法
即便这样，我们输出的数据仍然是不有效的（但是上面几点说明mask的方向是对的）我们为了让其能够更好的学习点的坐标，增加了更多的channel
|channel||
|--|--|--|--|--|--|--|--|--|
|x|[8,]|
|y|[8,]
|z|[8,]
|t|前四个点的厚度+后四个点的厚度
|mask|前四个点的mask+后四个点的mask
整个模型都要改了，开摆，明天再写，下班