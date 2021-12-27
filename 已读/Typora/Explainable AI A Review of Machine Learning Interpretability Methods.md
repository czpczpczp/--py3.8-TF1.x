# Explainable AI: A Review of Machine Learning Interpretability Methods

## 1 摘要和介绍

机器学习很有用，高风险场景要更可信，所以就有可解释性研究

## 2 摘取的部分

可解释性越强，越容易在系统的输入输出中识别因果关系，人们对模型内部过程理解越深。interpretable的解释性似乎是可理解即阐述点在模型，explainablity似乎是对产生的结果进行解释。不过文中也说这两者定义模糊很难界定

------

如果方法只提供了对特定实例的解释，那么它是局部的，如果该方法解释了整个模型，那么它是全局的。

可解释性方法可以分类的不同方面可视化如下图

![image-20211207161236727](C:\Users\10141\AppData\Roaming\Typora\typora-user-images\image-20211207161236727.png)

可解释性方法主要分为四类:解释复杂黑箱模型的方法，创建白箱模型的方法，促进公平和限制歧视存在的方法，以及分析模型预测敏感性的方法。（右上）

------

黑箱

事后可解释方法：这些方法并不是试图创建可解释的模型，而是试图解释已经训练过的，且通常是复杂的模型。

------

黑箱-深度学习模型专用的一些解释性方法【7-12】

1.梯度方法及其变种

2.反卷积方法可以用来可视化卷积神经网络

3.类激活映射（Class Activation Maps，CAMs）

4.分层关联传播（LRP）

5.RISE算法

下图为更具体的细分

![image-20211207165105869](C:\Users\10141\AppData\Roaming\Typora\typora-user-images\image-20211207165105869.png)

下图是不同方法的实例对比

![image-20211207165357339](C:\Users\10141\AppData\Roaming\Typora\typora-user-images\image-20211207165357339.png)

------

黑箱-解释任何黑箱模型的可解释性方法【12-18】

1.局部可解释模型-不可知解释(LIME)方法

2.加法解释(Shapley Additive interpretation, Shapley)，SHAP

3.Ancors方法，适用于任何具有高概率保证的黑箱模型

4.对比解释法(CEM)

5.反事实解释

6.Protodash方法

7.排列重要性（PIMP）

8.L2X方法

9.PDPs方法----->ICE图，ALE图

10.LIVE方法，LIVE和LIME类似

11.breakDown，类似于SHAP

12.ProfWeight

下面的图是更细致的分类

![image-20211207173055874](C:\Users\10141\AppData\Roaming\Typora\typora-user-images\image-20211207173055874.png)

![image-20211207173101466](C:\Users\10141\AppData\Roaming\Typora\typora-user-images\image-20211207173101466.png)

根据作者所说SHAP方法相对较好，因为对任何模型和任何类型 的数据提供解释，但也有缺点，它会高估不太可能的数据点。

------

白箱可解释性【18-19】

如线性回归，决策树之类的内部原理相对容易理解，有更好的可解释性。

白盒方法如下图

![image-20211208153352482](C:\Users\10141\AppData\Roaming\Typora\typora-user-images\image-20211208153352482.png)

------

机器学习公平性是机器学习可解释性的一个子领域，它通过评估它们的公正性和歧视来关注机器学习算法的社会和伦理影响。【19-27】

限制歧视和提高公平性的可解释性方法分类如下图

![image-20211208163127193](C:\Users\10141\AppData\Roaming\Typora\typora-user-images\image-20211208163127193.png)

公平性方法的可解释性是机器学习可解释性一个相对比较新的领域。

------

机器学习模型预测的敏感性分析的解释性方法

这些方法试图评估和挑战机器学习模型，以确保其预测是可信和可靠的。

------

传统敏感性方法【27-29】

![image-20211208165821713](C:\Users\10141\AppData\Roaming\Typora\typora-user-images\image-20211208165821713.png)

------

对抗性敏感性方法【29-35】

![image-20211208165932060](C:\Users\10141\AppData\Roaming\Typora\typora-user-images\image-20211208165932060.png)

![image-20211208165938751](C:\Users\10141\AppData\Roaming\Typora\typora-user-images\image-20211208165938751.png)

------

总结

在此分类下，可解释性方法被确定为四大类:解释复杂黑箱模型的方法、创建白箱模型的方法、促进公平和限制歧视存在的方法以及分析模型预测敏感性的方法。

因为深度学习火，所以可解释性文献大部分是神经网络的。（黑箱）

白盒的高性能模型很难创建

公平性相关的方法既不常见，也没有在占主导地位的机器学习框架内得到很好的推广

敏感性分析在过去的几年中得到了巨大的增长



尽管可解释性领域快速增长，可解释的人工智能仍然不是一个成熟和建立良好的领域，经常遭受缺乏正式形式和没有很好商定的定义。因此，尽管学术界已经开发了大量的机器学习解释性技术和研究，但它们很少构成机器学习工作流和管道的重要组成部分。



上面图片中的缩写

W 	Whit-Box/Interpretable Models
PH 	Post-Hoc
F 		Fairness
S 		Sensitivity
L 		Local
G 		Global
Agnostic 			Model Agnostic
Specific 			Model Specific
tab					 Tabular Data

img 			Image Data
txt 			Text Data
graph 		Graph Data
