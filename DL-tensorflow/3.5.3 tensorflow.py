import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

tf.compat.v1.disable_v2_behavior()              # 将tensorflow2降级到1的配套部分
tf.compat.v1.disable_eager_execution()

M = 2  # 输入数据维度
K = 3  # 分类数量
n = 100  # 每个分类的数据量
N = n * K  # 全部数据个数

#  生成数据
X1 = np.random.randn(n, M) + np.array([0, 10])
X2 = np.random.randn(n, M) + np.array([5, 5])
X3 = np.random.randn(n, M) + np.array([10, 0])

Y1 = np.array([[1, 0, 0] for i in range(n)])
Y2 = np.array([[0, 1, 0] for i in range(n)])
Y3 = np.array([[0, 0, 1] for i in range(n)])

X = np.concatenate((X1, X2, X3), axis=0)   # X和Y都是向下做联接，Y每个都循环一百次是为了和X对应的分类tn对应上
Y = np.concatenate((Y1, Y2, Y3), axis=0)

#  tensorflow部分
W = tf.Variable(tf.zeros([M, K]))
b = tf.Variable(tf.zeros([K]))

x = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, M])
t = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, K])
y = tf.nn.softmax(tf.matmul(x, W) + b)

# python乘法是对应元素相乘，示意图在笔记本最后一面
cross_entropy = tf.reduce_mean(-tf.compat.v1.reduce_sum(t * tf.compat.v1.log(y), reduction_indices=[1]))

train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# 如果y每一行最大的索引和t每一行对应的最大索引一致，就是分类正确，用tf.equal来检查
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))

# 定义小批量及其个数
batch_size = 50
n_batches = N // batch_size

# 随机梯度下降在每轮迭代都要打乱数据
init = tf.compat.v1.global_variables_initializer()     # 1.0的语法，我也不太清楚，好像是想用tf的变量就要在Session.run里
sess = tf.compat.v1.Session()
sess.run(init)


for epoch in range(20):
    X_, Y_ = shuffle(X, Y)  # 随机打乱

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        sess.run(train_step, feed_dict={
            x: X_[start:end],
            t: Y_[start:end]
        })


X_, Y_ = shuffle(X, Y)

classified = sess.run(correct_prediction, feed_dict={
    x: X_[0:10],
    t: Y_[0:10]
})

prob = sess.run(y, feed_dict={
    x: X_[0:10]
})

print("classified: ")
print(classified)
print()
print("output probability: ")
print(prob)