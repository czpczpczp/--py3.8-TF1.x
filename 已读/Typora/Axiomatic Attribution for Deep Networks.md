# Axiomatic Attribution for Deep Networks

本文研究的问题是，将深度网络的预测结果归因到输入的特征中。

本文提出做归因要满足两条公理

1.Sensitivity(a)

定义：如果对于所有仅在一个特征上具有不同取值的输入(input)和基线(baseline)，并且模型为两者给出了不同的预测。那么，那个不同取值的特征应该被赋予一个非0归因。若一个归因方法满足上面的要求，则称该归因方法满足*Sensitivity(a)*。

![image-20211213184801044](C:\Users\10141\AppData\Roaming\Typora\typora-user-images\image-20211213184801044.png)

2.Implementation Invariance

functionally equivalent：如果两个网络对所有的输入均有相同的输出，则称这两个网络functionally equivalent(即忽略了网络的实现)。

Implementation Invariance：一个归因方法对于两个functionally equivalent网络的归因总是一致的，那么称该归因方法满足Implementation Invariance。


作者提出的积分梯度法

![image-20211213185041333](C:\Users\10141\AppData\Roaming\Typora\typora-user-images\image-20211213185041333.png)

这个公式可以得到输入x的分量i的归因