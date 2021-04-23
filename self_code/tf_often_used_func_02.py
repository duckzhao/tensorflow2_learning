# tensor2.0常用函数的记录,包括使用GradientTape进行求导，one-hot编码，softmax归一化，梯度下降时的assign_sub自更新
# argmax，argmin，tf.equal->用于判断两个矩阵中相同元素数量的技巧
# tf.concat tensor拼接
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息
import tensorflow as tf
import numpy as np

# 1.使用tf.GradientTape 计算某个变量的梯度/偏导---计算时候都是返回的数字结果，不是带有公式的那种，求导结果也都是数字
# GradientTape是一个求导记录器，在 tf.GradientTape() 的上下文内，“所有的函数计算步骤”都会被记录以用于求导
# 如果要计算损失函数的倒数（梯度），就要把损失函数的自定义计算步骤，完整的写在tape内，才可以求导并且自更新
with tf.GradientTape() as tape:
    # 定义一个变量w，且赋予初值.变量与普通张量的一个重要区别是其默认能够被 TensorFlow 的自动求导机制所求导，因此往往被用于定义机器学习模型的参数
    w = tf.Variable(tf.constant(value=3.))  # 只有变量才可以被tape求导记录
    # 定义一个损失函数loss，为w平方
    loss = tf.square(w)  # 会返回一个当前点的数值型loss值
    # 计算loss函数对当前w值求偏导的结果---即当前loss函数关于w变量的梯度, target目标函数，sources求导变量
    grad = tape.gradient(target=loss, sources=w)    # w变量是和loss函数有关的变量，在tape记录的loss函数形式中对变量w求导
print('变量w=3.时在函数w^2中求导数的结果为：', grad)


# 对于“多个变量”同时“求偏导”的示例如下：
X = tf.constant(np.arange(1, 5).reshape(2, 2), dtype=tf.float32)
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))   # 计算结果为数值型tensor
w_grad, b_grad = tape.gradient(L, [w, b])   # 以列表形式传入变量，则返回多个偏导结果->代入一个变量的当前值，求另一个变量的当前点偏导值
print(L, w_grad, b_grad)


# tensorflow手写自定义的梯度下降及参数自更新,使用线性模型y=aX+b拟合以下数据
X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)
# 归一化
X = (X_raw - X_raw.min())/(X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min())/(y_raw.max() - y_raw.min())

# 将x，y转为tf张量
X = tf.constant(X)
y = tf.constant(y)
# 使用 tape.gradient(ys, xs) 自动计算梯度；
# 使用 optimizer.apply_gradients(grads_and_vars) 自动更新模型参数。---->也可以使用assign_sub完成
a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)  # 梯度下降变量

num_epochs = 10000  # 迭代轮数
# 优化器可以帮助我们根据计算出的求导结果,以一定的学习率更新模型参数，从而最小化某个特定的损失函数，具体使用方式是调用其 apply_gradients() 方法
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)  # 控制梯度下降更新方式的优化器
# 开始迭代训练
for index in range(num_epochs):
    # 使用tf.GradientTape记录loss函数的内容
    with tf.GradientTape() as tape:
        y_pred = a * X + b  # 预测值
        loss = tf.reduce_sum(input_tensor=tf.square(y_pred - y))    # 损失函数有了
    a_grad, b_grad = tape.gradient(target=loss, sources=[a, b])  # 偏导数有了    tape可以脱离with域作用，显得代码结构更清晰
    # 使用optimizer.apply_gradients优化器完成参数自更新,传入一个list，list中的元素是tuple->（变量的偏导数，变量）
    optimizer.apply_gradients(grads_and_vars=[(a_grad, a), (b_grad, b)])    # 原地更新，无需返回
print('优化后的参数分别为：', a.numpy(), b.numpy())

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

# 使用tf.concat完成tensor不同维度上的拼接,使用list传入要拼接的对象，axis指定拼接轴，0为第一维方向拼接，增加行数
a = np.arange(12).reshape((3, 4))
b = np.arange(12, 24).reshape((3, 4))
print(tf.concat([a, b], axis=0))