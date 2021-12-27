# DLIME A Deterministic Local Interpretable Model-Agnostic Explanations Approach for Computer-Aided Diagnosis Systems

虽然LIME和类似的局部算法因其简单性而受到欢迎，但随机扰动和特征选择方法会导致生成的解释不稳定，对于相同的预测，可能会产生不同的解释。在本文中，提出了一个确定性版本的LIME。我们利用聚类分层聚类(HC)来将训练数据分组，并利用K-Nearest Neighbour  (KNN)来选择正在解释的新实例的相关聚类，而不是随机扰动。在找到相关的聚类之后，在选择的聚类上训练一个线性模型来生成解释。【LIME对实例进行了随机扰动，有不确定性，	DLIME改进了生成数据的方法】

LIME是最早的局部可解释模型之一，它通过随机扰动生成实例周围的模拟数据点，并通过对扰动点的预测响应拟合稀疏线性模型来提供解释。

------

LIME示意图

![image-20211224110544309](C:\Users\10141\AppData\Roaming\Typora\typora-user-images\image-20211224110544309.png)

DLIME示意图

![image-20211224110558890](C:\Users\10141\AppData\Roaming\Typora\typora-user-images\image-20211224110558890.png)

