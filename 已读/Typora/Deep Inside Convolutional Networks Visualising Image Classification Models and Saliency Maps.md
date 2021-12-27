# Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps
这篇文章是梯度类方法的起始，说的是为啥能用梯度做可视化（我感觉也算是可解释性）

Sc(I)**代表图片I送入CNN得到的类别c分数**(由CNN分类层计算得到)

给定一个训练好的分类CNN和感兴趣的类别，可视化方法基于数值来生成图像(能够体现CNN所学习到的类别信息)。
$$
\arg \max _{I} S_{c}(I)-\lambda\|I\|_{2}^{2}
$$
这个公式的目的是生成图片，即找到一个图片I让c类分数最大。找图片的过程中是不动网络的，改的是图片，这也就是为啥最后看到的那些很奇怪的图片中凸显出的是做判定的区域。

使用的Sc（I）不经过归一化和soft-max，因为如果使用归一化和soft-max会导致训练过程中通过最小化其他类来提高Sc，而我们要关注的只是哪些地方影响C类本身的分数。

然后Sc（I）的定义是一阶泰勒
$$
S_{c}(I) \approx w^{T} I+ b
$$

$$
w=\left.\frac{\partial S_{c}}{\partial I}\right|_{I_{0}}
$$

