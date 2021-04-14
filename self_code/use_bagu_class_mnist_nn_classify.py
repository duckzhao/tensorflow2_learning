# 使用model class构建学习mnist数据集的神经网络模型---六步法

# 1.import
import tensorflow as tf


# 2.导入数据，并进行归一化预处理
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train/255., x_test/255.


# 3.使用class---继承tf.keras.Model定义model类，并且实例化
class mnist_model(tf.keras.Model):
    def __init__(self):
        super(mnist_model, self).__init__()
        # 在构造函数中初始化的每层网络结构
        self.flatten = tf.keras.layers.Flatten()
        # 两个全连接层
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        # 第二个全连接完成10分类的输出任务，10个神经元
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        # 将输入的28*28 拉直为1维
        y = self.flatten(inputs)
        # 将一维向量输入第一层网络 N*748, 748*128
        y = self.dense1(y)
        # 再继续传递 N*128, 128*10
        y = self.dense2(y)
        return y
# 顺便实例化出来一个model
mnist_model = mnist_model()


# 4.在model.compile中配置训练方法，优化器、loss函数、网络输出和label的比较类型
mnist_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['sparse_categorical_accuracy'])


# 5.在model.fit()中执行训练过程，告知训练集和测试集的输入值和标签、每个batch的大小（batchsize）和数据集的迭代次数（epoch）
mnist_model.fit(x=x_train, y=y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)


# 6.打印网络结构信息
mnist_model.summary()