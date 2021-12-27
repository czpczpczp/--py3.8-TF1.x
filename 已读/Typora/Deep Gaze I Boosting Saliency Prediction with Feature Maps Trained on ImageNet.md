# Deep Gaze I: Boosting Saliency Prediction with Feature Maps Trained on ImageNet
模型结构如下图

![image-20211213162756222](C:\Users\10141\AppData\Roaming\Typora\typora-user-images\image-20211213162756222.png)

Deep  Gaze  I的模型结构:首先对图像进行下采样并使用Krizhevsky网络进行预处理。模型中包含的各层的响应然后按比例扩大到最大的网络层的大小，并标准化到具有单位标准差。这个映射列表然后用高斯核函数线性组合和模糊。为了补偿中心固定偏差，增加了先验分布的估计。最后，对模型输出进行soft-max，得到二维概率分布。

这个文章的方法增加了图片的可解释性的信息量。