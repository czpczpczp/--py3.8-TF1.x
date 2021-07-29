import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()    # 把tensorflow2.0降级到1.0

w = tf.Variable(tf.zeros([2, 1]))           # 异或门输入为1x2，w设置为2x1
b = tf.Variable(tf.zeros([1]))              # x*w后是一个标量，偏置b设置为一维标量

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])       # 1.0的语法，x现在还是空壳，用于接收之后输入的1x2的X
t = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])       # t不是0就是1，设置为接收之后输入的Y
y = tf.nn.sigmoid(tf.matmul(x, w) + b)                          # y为sigmoid函数，变量设置为 （x * w） + b

cross_entropy = -tf.reduce_sum(t * tf.compat.v1.log(y) + (1 - t) * tf.compat.v1.log(1 - y))   # 直观的交叉熵误差函数

train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)  # train_step用学习率为0.1 的梯度下降法去求最小化交叉熵误差函数成立时的w和b

correct_prediction = tf.equal(tf.compat.v1.to_float(tf.greater(y, 0.5)), t)   # y>=0.5用tf.greater输出1，反之为0，之后转为float，然后用equal和标签t进行对比，结果输出给correct_prediction
                                                                                #  为什么＞＝0.5呢？因为输出值在训练后会更贴近真实值
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])         # 或门的四种输入情况

Y = np.array([[0], [1], [1], [1]])                      # 四种输入情况对应的真实输出，即标签

init = tf.compat.v1.global_variables_initializer()     # 1.0的语法，我也不太清楚，好像是想用tf的变量就要在Session.run里
sess = tf.compat.v1.Session()
sess.run(init)

for epoch in range(200):                    # 迭代两百次
    sess.run(train_step, feed_dict={
        x: X,
        t: Y
    })

classified = sess.run(correct_prediction, feed_dict={    #检测y的学习效果，看y的输出是否和t对应
    x: X,
    t: Y
})

print(classified)

prob = sess.run(y, feed_dict={
    x: X,
    t: Y
})

print(prob)

print('w: ', sess.run(w))
print('b: ', sess.run(b))

