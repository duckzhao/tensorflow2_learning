# 本节中的参数详解，或者函数详解等等，参数含义、如何选取等都可以参见github中的详细笔记：
# https://github.com/dxc19951001/Study_TF2.0/blob/master/tensorflow2.md

# 使用八股，六步法搭建神经网络分类

# 第一步：导入相关模块
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import datasets
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息


# 第二步：指定输入网络的训练集和测试集，x_train和y_train， 也可以顺便拆分导入xtest ytest
# 导入鸢尾花数据集
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target
# 不区分训练集和测试集，在此仅打乱顺序即可,为了保证两次shuffle打乱的顺序前后标签一致，两次打乱前后设置相同的seed
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
# 如果想要保证复现，可以也指定tf中一些随机操作的种子,tf在训练时也会有随机数的产生，因此提前指定，指定一次，全局固定
tf.random.set_seed(116)


# 第三步：逐层搭建网络结构，model = tf.keras.models.Sequential()
# sequential([网络结构])实际上就是个容器，这个[]里面封装了一个神经网络结构
model = tf.keras.models.Sequential([
    # 目前这个网络中只有一个全连接层
    # 全连接层：tf.keras.layers.Dense(units=神经元个数，activation= "激活函数“，kernel_regularizer=哪种正则化) 注意l2带括号，为方法
    tf.keras.layers.Dense(units=3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
])


# 第四步：在model.compile()中配置训练方法，选择训练时使用的优化器、损失函数和最终评价指标。
# 主要是配置如何反向传播，梯度下降，优化参数
# 配置第三步中实例化出来的网络模型在bp优化矩阵参数时   选择的优化器optimizer和损失函数loss
# 以及网络输出结果 和 训练时输入的lable之间的关系 Metrics---，一般输入label都是一维向量，如果输出经过了softmax，则是类别*样本数 的二维向量。填入sparse_ categorical_accuracy
# 其余情况见github笔记
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),    # 如果输出不是原始输出都False，如softmax或者relu、onehot等
              metrics=['sparse_categorical_accuracy'])


# 第五步：在model.fit()中执行训练过程，告知训练集和测试集的输入值和标签、每个batch的大小（batchsize）和数据集的迭代次数（epoch）
# validation_freq参数指定每迭代多少次后，将验证集代入当前训练的模型参数，进行一次验证，看看当前训练的效果如何，会输出验证准确率和loss
# validation_split，指定从x_data，x_data中分割多少出来作为测试集，和下面的参数二选一即可。
# validation_data=(x_test, y_test)，如果之前手动拆分了训练集和测试集数据，则通过次参数传入tuple指定验证集即可，这样就不会从xy中提取测试集了
model.fit(x=x_data, y=x_data, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)


# 第六步：使用model.summary()打印网络结构，统计求解的参数数目
model.summary()


# 随后开始run，tf框架会自动的开始训练网络结构，并打印各种信息！包括每一轮训练时的loss，acc和每20轮代入测试集验证的loss和验证准确率