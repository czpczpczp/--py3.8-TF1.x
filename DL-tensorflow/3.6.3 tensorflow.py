import tensorflow as tf
import numpy as np

tf.compat.v1.disable_v2_behavior()              # 将tensorflow2降级到1的配套部分
tf.compat.v1.disable_eager_execution()

#  定义数据,异或门
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

#  x和t是输入部分数据
x = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, 2])
t = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, 1])

#  输入层到隐藏层的定义，定义要训练的变量和函数表达式
W = tf.Variable(tf.compat.v1.truncated_normal([2, 2]))
b = tf.Variable(tf.zeros([2]))
h = tf.nn.sigmoid(tf.matmul(x, W) + b)

#  隐藏层到输出层的定义
V = tf.Variable(tf.compat.v1.truncated_normal([2, 1]))
c = tf.Variable(tf.zeros([1]))
y = tf.nn.sigmoid(tf.matmul(h, V) + c)

#  定义误差函数，二分类加上激活函数为sigmoid函数的交叉熵误差如下
cross_entropy = - tf.reduce_sum(t * tf.compat.v1.log(y) + (1 - t) * tf.compat.v1.log(1 - y))

#  定义随机梯度下降法，即在完成了变量和式子以及误差函数的定义后，再定义优化方法、
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

#  预测是否正确判断
correct_prediction = tf.equal(tf.compat.v1.to_float(tf.greater(y, 0.5)), t)

#  运行对话和变量初始器
init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)

#  迭代部分
for epoch in range(4000):
    sess.run(train_step, feed_dict={
        x: X,
        t: Y
    })
    #  每一千次迭代输出一次进度
    if epoch % 1000 == 0:
        print('epoch: ', epoch)

#  验证部分
classified = sess.run(correct_prediction, feed_dict={
    x: X,
    t: Y
})

prob = sess.run(y, feed_dict={
    x: X
})

print("classified: ")
print(classified)
print()
print('output probability: ')
print(prob)

#  tensorflow过程为定义输入输出，定义变量，定义激活函数，定义误差函数，定义训练方法，定义预测方法，初始对话和变量初始器，迭代训练用feed_dict喂数据，然后验证部分喂数据看参数训练效果
