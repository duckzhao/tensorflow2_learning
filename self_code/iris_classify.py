# 实现鸢尾花数据的分类
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息
import pandas as pd
from sklearn import datasets
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# 导入鸢尾花数据集，是以numpy格式存在的
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target
print(x_data.shape, y_data.shape)

# 给x_data 加上列（属性）索引，转为dataframe格式
all_data = pd.DataFrame(data=x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'])
pd.set_option('display.unicode.east_asian_width', True)  # 设置列名和数据列对齐，否则列名太长了
print(all_data)
# 给x_data 加上labels值，dataframe和dict有点像，可以直接新增key-value（最好新增列长度和以前一样长）
all_data['类别'] = y_data
print(all_data)

# 打乱原始的数据集
np.random.seed(116)  # 设置随机数种子，先打乱样本数据
np.random.shuffle(x_data)  # 原地修改，不需要接收返回值
np.random.seed(116)  # 设置同样的随机数种子，打乱标签列，保证样本和标签打乱前后是匹配的
np.random.shuffle(y_data)

# 从原始数据取出训练集和测试集
x_train = x_data[0:-30]
y_train = y_data[0:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]
print(x_train)    # 120,4

# 这个很重要，基本上外部读入的数据都要经过这一步骤
# 如果将这个x_train直接导入和w1矩阵相乘，会报类型错误，因此强转下类型，不强转走到后面是float64，强转了走到后面是float32
x_train = tf.cast(x_train, dtype=tf.float32)
x_test = tf.cast(x_test, dtype=tf.float32)
# print('x_train:\n', x_train)

# 将输入特征和标签进行配对，并指定训练时每次喂入的batch
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 定义神经网络中所有可训练的参数
w1 = tf.Variable(initial_value=tf.random.truncated_normal(shape=[4, 3], mean=0, stddev=0.1, dtype=tf.float32, seed=1))
b1 = tf.Variable(initial_value=tf.random.truncated_normal(shape=[3, ], mean=0, stddev=0.1, seed=1))
# print(w1)

# 定义训练中的一些超参数
lr = 0.1    # 学习率
train_loss_result = []  # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
test_acc = []  # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
epoch = 500  # 循环500轮，即迭代数据集500次
loss_all = 0  # 每轮分4个step，loss_all记录四个step生成的4个loss的和


# 循环训练逻辑
for epoch in range(epoch):
    # 每一个小循环数据集就已经被迭代了一次了
    for step, (x_train, y_train) in enumerate(train_db):    # batch级别的迭代，如果from_tensor_slices时不指定batch就是一个一个样本迭代，此时130个样本一共迭代130/32=4次
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1  # 神经网络前向过程中的主要计算  32*4 4*3 -> 32*3  + 1*3(会给每一行都加上这个偏置，等效于32*3的b1) 得到预测值
            # print(y)    # y大小为32*3 ，且返回值，有负，有大于1，不符合概率分布，因此softmax很有必要
            y = tf.nn.softmax(y)    # 将输出的y进行归一化，转换为概率分布    32*3 -> 32*3，转换后符合概率分布  得到预测概率
            y_ = tf.one_hot(y_train, depth=3)   # 将32*1一维向量转为 32*3 的one-hot矩阵
            loss = tf.reduce_mean(tf.square(tf.subtract(y_, y)))  # reduce_mean参数为 32*3 矩阵，由于reduce_mean没指定axis参数，对全部值求平均，返回数值()tensor对象loss
            loss_all += loss.numpy()  # 将每个step计算出的loss累加，为后续求loss平均值提供数据
            # 要求偏导进行更新的参数集合
            variables = [w1, b1]
            grads = tape.gradient(loss, variables)   # 对loss这个公式中的代入变量 w1， b1求偏导，计算分别梯度 ，返回(此处w1偏导方向的值4，3，b1偏导方向的值 3 )
            # 根据计算出的梯度下降方向，进行梯度更新，修改w1和b1的值
            w1.assign_sub(lr*grads[0])  # lr*grads[0](4, 3)的梯度值
            b1.assign_sub(lr*grads[1])  # lr*grads[1](3,)的梯度值

    # 每个epoch，打印一次loss信息---实际上一个epoch已经训练完了所有样本，后续一直在重复样本训练
    print("Epoch {}, loss {}".format(epoch+1, loss_all/4))  # 因为上面的loss_all是累积的一个epoch中4次batch的loss，因此/4，求平均，使得loss值更具有代表性
    train_loss_result.append(loss_all/4)    # 记录每个epoch的训练loss，后续绘制loss曲线，观察随着epoch增多，loss下降
    loss_all = 0    # 变量置0

    # 测试部分，每一个epoch都用测试集测试一次训练效果
    # 测试集中预测正确的数量，全部测试集数量
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:  # 一次取32个，但是因为测试集只有30个，因此取出来了30个
        # 使用本epoch更新后的w和b完成预测
        y = tf.matmul(x_test, w1) + b1  # 30*4 4*3 -> 30*3
        y = tf.nn.softmax(y)    # 转为符合概率分布的y 30*3
        pred = tf.argmax(y, axis=1)  # 返回每行预测概率最大的（列）索引 30*1
        # print(pred)
        # 将pred转换为y_test的数据类型,因为后面要用equal比较，因此要把上一个argmax返回的int64的索引号变为int32
        pred = tf.cast(pred, dtype=y_test.dtype)    # y_test.dtype 为int32  30*1
        # print(pred)
        # 比较预测结果和实际样本标签, 返回值为等大小的bool tensor
        correct = tf.equal(pred, y_test)    # 30*1
        # 将bool强转为int32，以便求和    true->1 false->0
        correct = tf.cast(correct, dtype=tf.int32)  # 30*1
        correct = tf.reduce_sum(correct)    # 1
        # 累加每个循环一共预测正确的样本数，但是这个循环只进行一次，其实不需要
        total_correct += correct.numpy()
        # 一共预测总共样本数，固定只有30个
        total_number += x_test.shape[0]
        # print('aaaaaaaaaaaaaaaaaaaaaaaaa')      # 每一个epoch确实只打印一次这个，因此说明该循环仅进行一次
    # 当这一次循环跑完了，就表示测试集预测结束，实际循环之跑一次
    # 本次预测的正确率
    acc = total_correct/total_number
    # 记录该正确率用于后续绘制正确率图像
    test_acc.append(acc)
    print('test acc is: ', acc)
    print('-'*30)

# 当所有训练，预测都结束后开始绘制图像
# 绘制loss曲线
plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss Score')
plt.plot(train_loss_result, label="$Loss$")  #$表示斜体     # x可以省略默认从0 1 2 。。。N-1递增
plt.legend()  # 画出曲线图例
plt.show()  # 画出图像

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()