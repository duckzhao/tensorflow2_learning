# 生成常量-张量的各种方法记录

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息
import tensorflow as tf
import numpy as np

# tf.constant 表示创建一个张量/常量，其原型api如下
'''
tf.constant(
    value,  # 传入的值，可以传入list或者数值（数值则shape不传入任何list）
    dtype=None, # 生成张量的类型，一共有 tf.int32, tf.float32, tf,float64等等，不指定dtype时整数默认int32，小数默认float32,如果value和dtype类型不一样会报错
    shape=None,  # 生成张量的大小，传入一个list ，表明张量矩阵大小
    name='Const',   # 生成张量的名字，不指定名字时系统应该会默认分配一个随机名字
    verify_shape=False， # 是否验证value和shape大小匹配， 一般不用传入
)

张量一共有三个重要的属性：A.shape， A.dtype， A.numpy()->打印出值
'''

# 1.生成数值型张量，其shape为空
one_number = tf.constant(value=1, dtype=tf.int64, name='one_number')
print('数值型张量的结果是：', one_number)

# 2.生成一维张量（矩阵），当不指定dtype时,会自动根据value进行矩阵解析，其shape为(6, )，逗号隔开一个数字表明一维，6表明一维有6个元素
one_dismenson = tf.constant(value=[1, 2, 3, 4, 5, 6], dtype=tf.int32)
print('一维张量的结果是：', one_dismenson)

# 3.生成二维张量
# 如下list解析结果为（2， 3）的shape
two_dismenson = tf.constant(value=[[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
print('二维张量的结果是：', two_dismenson)
#或者以这种方式生成,注（arange是左闭右开区间，从0开始，第三个参数为步长）
two_dismenson = tf.constant(value=np.arange(6), shape=(2, 3))
print('二维张量的结果是：', two_dismenson)

# 4.生成固定维数的 某数字 填充张量矩阵
full_one_number = tf.constant(value=5, shape=(2, 3))
print('全5填充矩阵结果是：', full_one_number)
# 也可以使用tf.fill完成该功能, 但不用shape指定维数，而是dims指定，如果是一维第一个参数直接写个数即可, 无法指定dtype
full_one_number = tf.fill(dims=[2, 3], value=5.)
print('全5填充矩阵结果是：', full_one_number)

# 5.创建全为0或者全1的张量，[2, ]表示一维两个0，效果等于 2
full_zeros = tf.zeros(shape=[2, ])
print('全为0的一维张量：', full_zeros)
full_zeros = tf.zeros(shape=2)
print('全为0的一维张量：', full_zeros)
# 打印结果开始有几个[[[ 表明是几维的矩阵
full_ones = tf.ones(shape=[2, 3, 4])
print('全为1的三维张量：', full_ones)

# 6.将其他数据类型转换为tensor类型， 以numpy产生的矩阵为例
# arange仅有reshape函数改变矩阵维数，初始声明仅产生一维矩阵
a = np.arange(6).reshape(2, 3)
# 该函数仅有三个参数，转换前后维数保持一致
two_dismenson = tf.convert_to_tensor(value=a, dtype=tf.float32, name='two_dismenson')
print('二维张量的结果是：', two_dismenson)


# 7.生成符合正态分布的随机数张量
# 其中shape，均值和标准差为最重要的三个参数，默认生成(0, 1)分布的正态分布矩阵
random_normal = tf.random.normal(shape=[2, 3], mean=0, stddev=1.0, dtype=tf.float32, seed=222, name='random_normal')
print('符合正态分布的随机数张量', random_normal)
# 还有一种阶段式正态随机数，生成的随机数范围在 均值+—2倍标准差 范围内,这种随机数更加集中在均值附近
truncated_random_normal = tf.random.truncated_normal(shape=[2, 3], mean=0, stddev=1.0, dtype=tf.float32, seed=222, name='truncated_random_normal')
print('符合正态分布的截断式随机数张量', truncated_random_normal)


# 8.生成均匀分布随机数,参数如下，其中前三个参数最重要
random_uniform = tf.random.uniform(shape=[2, 3], minval=0, maxval=1.0, dtype=tf.float32, seed=222, name='random_normal')
print('符合均匀分布的随机数张量', random_uniform)

# 9.将tensor转为numpy格式矩阵，不能改变维度和类型
a = random_uniform.numpy()
print('将tensor转为numpy：', a)