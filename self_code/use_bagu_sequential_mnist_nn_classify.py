# 使用sequential构建学习mnist数据集的神经网络模型---六步法

# 1.导入相关包
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息

# 2.指定训练集和测试集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 对输入特征进行归一化，到0-1之间，更有利于网络吸收
x_train, x_test = x_train/255., x_test/255.

# print(x_train.shape)
# print(x_test.shape)
# print(y_test[0])


# 3.逐层搭建网络结构
model = tf.keras.models.Sequential([
    # 拉直层，不进行任何计算，不需要传入参数
    tf.keras.layers.Flatten(),
    # 第一层，全连接层dense,128个神经元，该层输入  N*784，784，128=N*128
    tf.keras.layers.Dense(units=128, activation='relu'),
    # 第二层，用于将上层输入转换为10分类的输出概率，所以需要有10个神经元，每个神经元输出对应一个数字的概率，那个高表示预测结果是那个数字
    # 也就是 N*128 * 128*10 = N*10,再用softmax把它转换为输出概率即可
    tf.keras.layers.Dense(units=10, activation='softmax')
    ])


# 4. 在model.compile中配置训练方法，优化器、loss函数、网络输出和label的比较类型
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])


# 5.在model.fit()中执行训练过程，告知训练集和测试集的输入值和标签、每个batch的大小（batchsize）和数据集的迭代次数（epoch）
model.fit(x=x_train, y=y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)


# 打印网络结构信息
model.summary()