# tensor2.0常用函数的记录,包括使用GradientTape进行求导，one-hot编码，softmax归一化，梯度下降时的assign_sub自更新
# argmax，argmin，tf.equal->用于判断两个矩阵中相同元素数量的技巧
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息
import tensorflow as tf
import numpy as np

# 1.使用tf.GradientTape 计算某个变量的梯度
with tf.GradientTape() as tape:
    # 定义一个变量w，且赋予初值
    w = tf.Variable(tf.constant(value=3.))
    # 定义一个损失函数loss，为w平方
    loss = tf.square(w)
    # 计算loss函数对当前w值求偏导的结果---即当前loss函数关于w的梯度, target目标函数，sources求导变量
    grad = tape.gradient(target=loss, sources=w)
print(grad)


# 2.one_hot编码，主要用于将label列，转换为矩阵形式表示，label所在编码值为1，其余值为0，矩阵每行仅有1个1，表示当前label
classes = 3  # 编码深度/类别数
labels = tf.constant([1, 0, 2])  # 当前样本数据的label
# axis指定编码的轴，dtype指定编码的类型，默认为float32，不传入默认以行进行one hot编码，这样写就够用了
output = tf.one_hot(indices=labels, depth=classes)
print('one-hot编码的结果为', output)


# 3.tf.nn.softmax(x)把一个N*1的向量（该向量常为n分类任务的n个类别输出值）归一化为（0，1）之间的值，并和为1
# 由于其中采用指数运算，使得向量中数值较大的量特征更加明显
y = tf.constant(value=[1.01, 2.53, -0.86])
# logits要归一化的向量值， axis 当logits维数较多时，指定归一化方向，一维不用管
print('softmax后的输出值为', tf.nn.softmax(logits=y, axis=0))


# 4.variable的自更新操作---assign_sub,assign_add，自减和自加并更新结果自动赋值给原变量
# 如先指定变量w，这样w才是可以更新的, w可以是数值，也可以是矩阵，只要和sub操作的shape相同即可运算
w = tf.Variable(initial_value=[3., 2.], name='w')
# 指定w自减1, 操作结束后w的值为2.
w.assign_sub([1, 1])
print(w)


# 5.tf.argmax 和 tf.argmin 返回张量沿指定维度最大/小值的“索引”， axis=0 获取每一列最大值的索引，结果数=列数；axis=1 获取每一行最大值的索引
test = np.arange(12).reshape([3, 4])
print('生成的矩阵为：', test)
print('每一列最大值的索引为：', tf.argmax(input=test, axis=0))  # 索引从0开始
print('每一行最大值的索引为：', tf.argmax(input=test, axis=1))


# 6.tf.equal 判断两个规格，维度 完全相同的 矩阵（张量，numpy类型，list）是否相等  它是逐个元素比较判断的！！！
# 返回一个 与输入等维度 的矩阵，其中元素全变为bool类型。相等就是True，不相等，就是False
a = np.arange(12).reshape((3, 4))
b = np.arange(12, 24).reshape((3, 4))
print(tf.equal(a, b))

# 如果要统计相等的个数，可以使用cast将bool强转为int32，然后使用reduce_sum
print(tf.reduce_sum(tf.cast(x=tf.equal(a, b), dtype=tf.int32)).numpy())
