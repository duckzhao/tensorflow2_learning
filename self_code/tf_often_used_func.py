# tensor2.0常用函数的记录
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息
import tensorflow as tf
import numpy as np

# 1.tf.cast 强制tensor转换为某种数据类型
x1 = tf.constant(value=np.arange(12), dtype=tf.float32, shape=[3, 4])
print(x1)
# cast函数仅有三个参数，输出的shape和原来的shape保持一致
x2 = tf.cast(x=x1, dtype=tf.int32, name='int_list')
print(x2)


# 2.tf.reduce_min和tf.reduce_max 计算张量维度上元素的最大/最小值
x3 = tf.constant(value=np.arange(0, 6), shape=[2, 3])
print(x3)
x4 = tf.reduce_min(input_tensor=x3)
x5 = tf.reduce_max(input_tensor=x3)
print(x4)
print(x5)
# 实际上该函数是还可以传，axis，keepdims，name等三个参数，似乎返回值是tensor的函数都可以传name参数
# axis=0 获取每一列的xx值,结果数=列数 ；axis=1 获取每一行的xx值，结果数=行数;当不传axis时表示所有元素都参与运算，返回所有元素的xx值
# tf.reduce_mean和tf.reduce_sum 计算张量维度上元素的平均值/和 计算会以传入tensor的类型运算---如果传入int32平均值默认int32，会舍弃小数，所以做平均之前要强转int为float
x6 = tf.reduce_mean(input_tensor=tf.cast(x3, tf.float32), axis=0)
x7 = tf.reduce_sum(input_tensor=x3, axis=1)
print('每一列的平均值为：', x6)
print('每一行的和为：', x7)


# 3.tf.Variable 将变量标记为“可训练”，被标记的变量会在反向传播中记录梯度信息。神经网络训练中，常用该函数标记待训练参数
# 这个Variable可以传入tensor，或者python中的其他数据类型，但该数据类型要有shape属性
# 经常用随机正态分布生成初始 待训练变量，可以指定变量名，以最后一个生成函数指定的名字为准-w
w = tf.Variable(tf.random.normal([3, 4], mean=0, stddev=1, name='ww'), name='w')
print('生成一个变量：', w)


# 4.tensor的四则运算 --- 对应“元素”的 加减乘除 运算，只有维度相同的张量才可以做四则运算，并且最好都以dtype=float32进行运算
# ps 四则运算可以直接加减 x8-x9
x8 = tf.ones(shape=[2, 3])
x9 = tf.fill(dims=[2, 3], value=3.)

print('加法运算结果为：', tf.add(x8, x9))
print('减法运算结果为：', tf.subtract(x8, x9))
print('减法运算结果为：', x8-x9)
print('乘法运算结果为：', tf.multiply(x8, x9))
print('除法运算结果为：', tf.divide(x8, x9))


# 5.计算tensor的平方，n次方，开方---对每个元素执行操作
print('平方结果为：', tf.square(x9))
print('5次方结果为：', tf.pow(x9, 5))
print('开方结果为：', tf.sqrt(x9))


# 6.矩阵相乘 tf.matmul
# 生成一个 2x3 的矩阵
x10 = tf.constant(value=range(6), shape=[2, 3], dtype=tf.float32)
# 生成一个 3x2 的矩阵
x11 = tf.constant(value=range(7, 13), shape=[3, 2], dtype=tf.float32)
print('矩阵相乘的结果为：', tf.matmul(x10, x11))


# 7.切分传入张量的第一维度， 生成输入特征标签对，构建数据集
# data = tf.data.Dataset.from_tensor_ slices((输入特征，标签))
# (Numpy和Tensor格式都可用该语句读入数据)
features = tf.constant(value=np.arange(12), shape=[3, 4])   # 样本特征
labels = tf.constant(value=[1, 2, 3])   # 样本标签
dataset = tf.data.Dataset.from_tensor_slices((features, labels))    # 聚合为一个组合带标签样本---tensor格式的
print('构建的数据集为：', dataset)
'''
<TensorSliceDataset shapes: ((4,), ()), types: (tf.int32, tf.int32)>
表明每一个样本是一维长度为4的向量， 数值标签
'''
for element in dataset:
    print('样本的内容为：', element)