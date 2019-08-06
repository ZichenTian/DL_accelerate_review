# 紧致网络设计

## 1 手工设计

所谓紧致网络设计，简而言之就是用计算量更小的网络来代替VGG、ResNet等大网络，同时精度损失较小

目前的紧致网络大多使用depthwise-convolution/group-convolution，来达到减少conv3x3计算量的目的，同时使用conv1x1/shuffle等方式来融合各个通道间的特征

### 1.1 Squeezenet

论文：[https://arxiv.org/abs/1602.07360](https://arxiv.org/abs/1602.07360)  
代码：[https://github.com/DeepScale/SqueezeNet](https://github.com/DeepScale/SqueezeNet)

该论文并没有过多关注加快网络计算速度，而是把重点放在了减少模型大小上。根据论文中作者的论述，减少模型大小有三个好处：
1. 分布式训练时传递的参数更少，速度更快
2. 模型更小，更新起来更方便，尤其是OTA更新(over-the-air updates)
3. 在FPGA/ASIC/嵌入式端更友好

该论文的核心是设计了一个叫Fire Module的模块，来减少参数量/计算量：  
![DeepinScreenshot_select-area_20190805152213.png](https://i.loli.net/2019/08/05/1XzWwiLpF4e85kU.png)

1. 一开始的conv1x1用来减少channel数
2. 后面分为conv1x1和conv3x3，其中conv3x3用来保持感受野
3. 将conv1x1和conv3x3得到的特征融合起来

上述的1、2操作可以有效减少3x3卷积的参数量和计算量（输入channel少了，输出channel也少了），并保证了网络具有不错的精度

网络架构如下：  
![squeezenet_macroarchitecture.png](https://i.loli.net/2019/08/05/D1dPnmGETt539yq.png)

网络具体配置如下：  
![squeezenet_table1.png](https://i.loli.net/2019/08/05/iFdImT7UsHDroaQ.png)

网络精度如下：  
![squeezenet_result.png](https://i.loli.net/2019/08/05/pLqYmMC24NBvnOa.png)

可以看到，在参数量为AlexNet的1/50时，top-1 acc相对于AlexNet略有提升，top-5 acc精度一样，效果还是不错的；
作者还使用了Deep Compression的方法进行了量化，精度不减且模型体积更小

下表是SqueezeNet使用bypass后的精度，可以看到，使用直接的连接比加卷积再bypass的效果要好：  
![squeezenet_result2.png](https://i.loli.net/2019/08/05/Uz63glvMmpywdDP.png)

值得一提的是，操作2中1x1和3x3卷积output channel是可以根据需要变化的。如果想要效果好一点，可以增加conv3x3的输出channel数；如果想要速度快、参数量少的话，可以减少conv3x3的输出channel数，这点在论文第五章有详细描述

**总结：**  
本篇文章之所以能够做到精度不减，不仅仅得益于Fire Module的设计，还吸收了如下idea：
* 深的网络比较好 from VGG
* bypass效果比较好 from ResNet
* late-downsample效果好 from PReLU

同时，除了Fire Module之外，论文还使用了NiN提出的Global Average Pooling替代FC的方式，也减少了一定的参数量  

作者没有给出速度上的对比，实属遗憾。不过根据实际应用情况来看，SqueezeNet确实能够跑的很快

### 1.2 MobileNet V1

论文：[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861v1)  
代码：  

该篇文章的核心思想是使用depthwise-conv3x3 + pointwise-conv1x1，来替代原来大计算量的conv3x3：  
![mobilenetv1_module.png](https://i.loli.net/2019/08/05/I2W5VGt1SgTBwCo.png)

depthwise-conv3x3能够大大节省计算量和参数量，同时保持感受野；
pointwise-conv1x1能够“打通”各个channel，弥补depthwise-conv各个channel之间特征不融合的缺点  
这两个算子的融合能够节省多少计算量这里就不再推导了，网上一搜一大把

网络一共28层，结构如下：  
![mobilenetv1_network.png](https://i.loli.net/2019/08/05/dJ2vLwce1trz8Tq.png)

根据文章中的叙述，MobileNetV1 95%的计算花在conv1x1上，当时卷积基本上都是用im2col+gemm的方式算的，而conv1x1并不需要im2col，直接gemm即可，所以速度很快：
![mobilenetv1_ops.png](https://i.loli.net/2019/08/05/824pAduSVUFDbMj.png)

下图是MobileNetV1和其他网络的对比，可以看到MobileNetV1的效果还是很好的：  
![mobilenetv1_result.png](https://i.loli.net/2019/08/05/K9FRUYiymnJgrCd.png)

在MobileNetV1的基础上，还可以通过减少输入图片的分辨率、减少每层的channel数来进一步压缩模型

**总结：**  
MobileNetV1是一篇不错的文章，读起来让人感觉很舒服  
但值得一提的是，论文在文字表述中强调95%的时间花在conv1x1上，这是不正确的。事实上，depthwise-conv3x3是一种很慢的卷积，不可能只占5%的时间  
*TODO：待测试*

另外，文章在Detection和Face Recognization上也做了验证性的工作，十分solid  
不过模型本身是sequence模型，梯度反传比较难

### 1.3 MobileNet V2

论文： [MobileNetV2：Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)  
代码：  

MobileNetV2的核心仍然在于Block上，论文称其为Inverted Residual Block：  
![mobilenetv2_module.png](https://i.loli.net/2019/08/05/EKLHIDtRUpvX9eF.png)  
![mobilenetv2_module2.png](https://i.loli.net/2019/08/05/1CM7tqQ4fWYjTmy.png)

该module主要分如下四步： 使用conv1x1升维（增加channel数） -> dwconv3x3 -> conv1x1降维（不加relu） -> eltwise-add shortcut

其相对于V1的核心改进在于：  
1. 使用了残差结构来强化梯度反传
2. 使用Inverted Residual Block。和ResNet的Residual Block的降维->conv3x3->升维不同的是，该block是升维->做dwconv3x3->降维，是一个纺锤形；且最后降维的时候，是不加ReLU的:
![mobilenetv2_block_compare.png](https://i.loli.net/2019/08/05/9AftJqCkLx8e4vj.png)  

这么做的目的在于：  
1. 在较高维度下做dwconv3x3效果比较好  *TODO：再看看，没看懂*
2. 在较高维度下，ReLU在引入非线性的同时不会造成太大的精度损失，如下图所示：  
![mobilenetv2_relu.png](https://i.loli.net/2019/08/05/28VKWemLpjQws3M.png)  
可以看到，图案在映射到高维空间做ReLU，再映射回来后，形状变化不大；而低维空间这么做形状变化很大。可见在高维空间下，ReLU造成的信息损失更小  
3. 同上，由于ReLU在低维会带来较大的损失，因此在最后降维时，不进行ReLU

网络结构如下：  
![mobilenetv2_network.png](https://i.loli.net/2019/08/05/uZq1SgrHVX7C8YG.png)  

ImageNet上的结果如下，可以看到V2比V1有1.4个点的提升，计算量小了近一半：  
![mobilenetv2_result.png](https://i.loli.net/2019/08/05/a39bkKWcX1GE7xL.png)

**总结：**   
简而言之，MobileNetV2在V1的dwconv3x3+ptwise1x1的基础上，增加了Residual link和升维->降维，减少计算量的同时大大提升了效果

这里要说一下，V2看似升维增加了channel，但实际上每个block的基础channel数是小于V1的，因此整体计算量也是小于V1的（可以对比上文中的表格）

这里再次讨论一下Inverted Residual Block和ResNet中的Residual Block的区别，先借用一张图：  
![mobilenetv2_resnet_compare.png](https://i.loli.net/2019/08/05/XZex41gm89yiuqw.png)  
可以看到，ResNet是先做了一个0.25x降维，再做的full conv3x3，再进行4x升维。这么做的目的在于节省conv3x3的计算量（因为毕竟是个完全卷积）  
而MobileNetV2由于使用的是depthwise conv 3x3，根本不担心计算量的问题，因此才敢考虑先升维（一般是升维6倍）以保证效果。  
理论上ResNet使用Inverted Residual Block也能获得很好的效果( *TODO：可以验证一下* )，但试想如果ResNet先升维6倍，计算量和参数量就太大了。所以说正是dwconv3x3才使得升维成为了可能  
*TODO：探究一下升维channel带来的性能收益和时间减益*

### 1.4 ShuffleNet V1

论文： [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)
代码： 

ShuffleNet V1的Block结构如下图所示：  
![shufflenetv1_block.png](https://i.loli.net/2019/08/05/cm5jF6DWapudAqz.png)
![shufflenetv1_theory.png](https://i.loli.net/2019/08/05/MreTjywvxiFYK2m.png)

其核心思路在于：  
1. 使用group conv 1x1 代替其他网络的full conv 1x1（即pointwise-conv），节省计算量
2. 由于group conv 1x1 无法有效融合各个group间的信息，ShuffleNet设计了shuffle操作，用于交换各个group的信息
3. 使用了Residual block来强化梯度反传

网络结构如下：  
![shufflenetv1_network.png](https://i.loli.net/2019/08/05/2i19E6GjwZn7y5t.png)

论文探究了g=1～g=8（group num）的效果，g越大，效果越好；  
论文还表示在ARM平台下，g=3可以达到精度和实际速度最好的trade-off；理论时间每降低4x，实际速度降低约2.6x左右  

同计算量下，shufflenet效果优于mobilenetV1等其他网络，这里就不放结果了，对比比较多

这里要对Shuffle操作再讨论一下：  
首先，shuffle操作是可导的，也就是说ShuffleNet本身是可以端到端训练的  
其次，shuffle操作在训练时是易于实现的，设tensor的channel分为g组，每组n个channel，则将该tensor拆分为(g,n)两个维度，再transpose为(n,g)，之后再展平为n\*g一个维度，即完成了shuffle；换句话说，shuffle操作并不是随机的，而是固定的  *TODO：验证随机是否好&&dropout是否好用*  
不过在测试时，transpose操作略微费时，猜测是跟后面的dwconv3x3合并为了一个算子  *TODO：验证该想法*  
另外，论文表示在一个block中，shuffle只进行一次就够了；shuffle、dwconv3x3、以及之后的gconv1x1并不需要ReLU。个人认为这是因为shufflenet和resnet一样，第一个gconv进行了下采样，使得后面的channel数很少，ReLU会带来较大信息损失

总结：  
ShuffleNet V1是一篇很好的文章，该文章通过GConv代替其他网络的full conv来进一步降低conv1x1的计算量，同时使用不产生计算量的shuffle操作来交换group间的信息，设计十分精巧且易于实现  
其中shuffle操作比较吃底层实现。底层实现的好，例如和后面的dwconv一块进行的话，就可以完全隐藏shuffle的时间；实现的不好，例如就是用reshape->transpose->reshape实现的话，则会产生额外的内存操作（分配和搬运），比较浪费  
另外，实际应用中，g=3和g=8用的比较多，一个是在ARM端trade-off，另一个是效果最好

### ShuffleNet V2

论文： [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)  
代码：  

ShuffleNetV2这篇论文首先阐明了两个观点：  
1. FLOPS并不能作为实际执行速度的指标，有很多因素会导致计算量和执行速度不成正比，如MAC和并行度
2. 相同的FLOPS也会因为执行平台和底层优化的不同而速度不同，例如low-rank方式速度并不快，主要原因是cudnn对conv3x3优化得太好了

所以作者表示，应当用实际执行速度作为唯一指标，而且测的时候一定要在目标平台上测试

作者紧接着提出了设计网络的四条建议：  
1. Conv的输入输出channel数应当相等（这是针对conv1x1来的，conv3x3有其最小值，不太一样）  *TODO：柯西不等式*
2. 作者表示，在FLOPs和输入shape不变的情况下，group越多，MAC越大，实测速度会慢很多，所以group越小越好  *存疑，觉得没有实际意义  TODO：固定输入输出shape进行测试*
3. 一些碎片化的层（Network Fragmentation）（其实就是计算量小的层）会使得网络并行度变差，对ARM还好，对GPU有害
4. Eltwise、ReLU等操作会占不少时间，虽然几乎没有多少计算量，但是是严重的访存密集型操作，GPU这种高性能平台对这种算子算不快，不应当忽略

作者表示之前的state-of-the-art模型违背了上述原则：  
shufflenetv1违背了原则2，其bottleneck结构违背了原则1  
mobilenetv2的inverted residual block违背了原则4  
自动生成的结构有很多碎片，违背了原则3

之后作者提出了ShuffleNetV2，其结构如下图cd，其核心思路是：
![shufflenetv2_block.png](https://i.loli.net/2019/08/06/X9L8qQvuGfrC1cZ.png)  
1. 使用channel_split，再concat的方式，而不是传统residual block的sum的方式，这样避开了Eltwise操作
2. 一个branch是直连，另一个branch是三个conv，输入输出channel数都一样
3. 不再使用Gconv，而是直接使用conv1x1，这样速度比较快，同时也不再需要shuffle了
4. channel_split + concat的方式和densenet有异曲同工之妙

*TODO：想想dwconv3x3有没有改进的余地*

网络结果：  
![shufflenetv2_result.png](https://i.loli.net/2019/08/06/ZGSDjCE6dhn9bM3.png)

总结：  
ShuffleNetV2是一篇非常好的论文，它第一个强调了FLOPS不能作为执行速度的依据，并从MAC、Eltwise等角度分析了如何实际执行速度，最后设计了高效的网络Block，兼顾了速度和精度，尽管FLOPS并不是最小的  
实际用起来shufflenetv2在GPU和CPU上都很快，以我个人的经验，在PX2上，MobileNetV2慢于ResNet18-half慢于ShuffleNetV2，可见MobileNetV2虽然计算量不大，但是确是很慢的，甚至慢于ResNet18-half；而ShuffleNetV2不愧是考虑到了实际执行速度而设计的网络，其速度明显快于MobileNetV2。

*TODO：补一组实验出来*

*TODO: Xception, CondenseNet， IGCV*
*TODO：写一篇ShuffleNetV2相似的论文*


## 2 AutoML方式设计

*TODO：认真查一下*
*Learning transferable architectures for scalable image recognition*
*Progressive neural architecture search*
*Regularized evolution for image classifier architecture search*
*NasNet-A*

## 参考文献

[Squeezenet: Alexnet-level accuracy with 50x fewer parametersand< 0.5 mb model size](https://arxiv.org/abs/1602.07360)  
[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications](https://arxiv.org/abs/1704.04861v1)  
[MobileNetV2：Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)  
[MobileNetV1 & MobileNetV2 简介](https://blog.csdn.net/mzpmzk/article/details/82976871)  
[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)  
[轻量级网络ShuffleNet v1](https://www.jianshu.com/p/29f4ec483b96)  

