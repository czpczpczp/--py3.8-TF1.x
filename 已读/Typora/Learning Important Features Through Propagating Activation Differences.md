# Learning Important Features Through Propagating Activation Differences

作者提出了深度学习重要特征(Deep Learning Important FeaTures)，一种新的算法，为给定输出的输入分配重要性分数。

之前的人研究的是正向添加扰动，作者说效果不好，作者的方法是反向传播方法，将重要信号从输出神经元向后通过各层传递到输入神经元，使其效率更高。DeepLIFT就是这样一种方法。

