# 使用Sequential搭建神经网络模型时，网络中隐层之间的关系只能是“上层的输出就是下层的输入”这样的顺序结构
# 无法写出一些带有跳连的非顺序网络结构，即对上层输出做一些处理后，再输入下层网络
# 为了解决这个问题，我们选择用类class搭建神经网络结构。

# 1.导入相关库
import tensorflow as tf
from tensorflow.keras import Model
from sklearn import datasets
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息

# 2.准备数据，样本和label
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target
# print(x_data, '\n', y_data)
# 打乱样本数据，防止样本数据排序过于规律
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 3.使用class方式，新建网络结构类
class iris_model(Model):
    # 在init中，定义所需要的网络块
    def __init__(self):
        # 调用父类的构造方法，实例化部分父类属性，即类的name
        super(iris_model, self).__init__()
        # 接下来定义各种所需网络块，网络块之间无直接联系,这里的网络定义api和sequential中一致，参数什么也都一致
        # 还是依旧只有一层全连接网络
        # softmax将输入映射为符合概率分布的 类别概率，而sigmoid是将输出映射为0-1之间的小数，所有类别和不一定为1，不过也是样本*类别 的二维矩阵
        self.dense1 = tf.keras.layers.Dense(units=3, activation='sigmoid', activity_regularizer=tf.keras.regularizers.l2())

    # 重写父类的call函数，在call函数中实现前向传播，通过在call中调用init函数里定义的网络结构，进行传参和返回值的接收
    # call函数接收的输入就是fit中指定的每次输入的batch_size
    # 再此过程中可以实现   对上一层的输出接收以后，先进行一些中间处理，再传入下一层网络运算
    def call(self, x):
        # 每一层网络运算完毕，会返回一个y
        y = self.dense1(x)
        return y

# 网络定义完毕之后，实例化一个model出来
model = iris_model()


# 4. 在model.compile中配置训练方法，优化器、loss函数、网络输出和label的比较类型
# 实际上model.compile和model.fit中的设置，都是针对整个网络模型而言的，每一层传播中用的都一样，而单层网络中有区别的在class中已经独自定义好了
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              # label是数字，输出是向量
              metrics=['sparse_categorical_accuracy'])


# 5.在model.fit()中执行训练过程，告知训练集和测试集的输入值和标签、每个batch的大小（batchsize）和数据集的迭代次数（epoch）
model.fit(x=x_data, y=y_data, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)


# 打印网络信息
model.summary()