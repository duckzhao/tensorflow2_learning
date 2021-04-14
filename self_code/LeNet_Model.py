# 开始写baseline代码，之后的cnn网络都可以从这个模板中加以修改，数据源此时都用cifar10
# 这种网络模型 训练时对于 输入图像 的维度没有要求，但是验证时输入图像的维度必须和训练的维度保持一致
# Conv2D在执行卷积的过程中是可以对三通道图片进行处理的，使用的filters也会是三通道的，（可能要通过input_shape说明下才行）

# 1.
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
import numpy as np
np.set_printoptions(threshold=np.inf)

# 2.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train/255., x_test/255.
print('训练集x的维度：', x_train.shape, '结果y的维度：', y_train.shape)

# 3.
class LeNet(tf.keras.Model):
    def __init__(self):
        super(LeNet, self).__init__()
        # 当没有BN操作时时可以把激活直接写在卷积操作里 #

        # CBAPD---6个5*5的卷积核，卷积步长时1，不使用全零填充，不使用BN操作，sigmoid激活函数，2*2的池化核，步长为2，不使用全零填充，不使用Dropout
        self.conv1 = Conv2D(filters=6, kernel_size=(5, 5), strides=1, padding='valid')  # 卷积层
        # self.bn1 = BatchNormalization()  # 不用写参数，但是会多计算几个变量，标准化卷积完了输出至正态分布时的变量,一个卷积核多2个变量
        self.act1 = Activation(activation='sigmoid')  # 激活层
        self.pool1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')  # 池化层
        # self.drop1 = Dropout(0.2)  # dropout层

        # 16个5*5的卷积核，步长1，valid填充，不使用BN，sigmoid激活函数，最大池化，2*2，步长2，valid填充，不使用Dropout
        self.conv2 = Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding='valid')
        self.act2 = Activation(activation='sigmoid')
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')

        self.flatten = Flatten()

        self.dense1 = Dense(units=120, activation='sigmoid')
        self.dense2 = Dense(units=84, activation='sigmoid')
        self.dense3 = Dense(units=10, activation='softmax')


    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        y = self.dense3(x)
        return y

baseline_model = LeNet()

# 4.
baseline_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                       metrics=['sparse_categorical_accuracy'])

# 5.
# 再fit之前设置保存model，并且判断能否加载历史训练参数
check_point_path = './check_point/LeNet.ckpt'
if os.path.exists(check_point_path+'.index'):
    print('-------------load the mode1-------------')
    baseline_model.load_weights(check_point_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_point_path, save_weights_only=True, save_best_only=True)

history = baseline_model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test),
                             validation_freq=1, callbacks=[cp_callback])

# 6.
baseline_model.summary()

# 打印参数名称，并保存数值参数到本地
# print(baseline_model.trainable_variables)
with open('./cifar10.txt', 'w')as f:
    for tensor_variable in baseline_model.trainable_variables:
        print(tensor_variable.name)
        print(tensor_variable.shape)
        # print(tensor_variable.numpy())
        f.write(str(tensor_variable.name) + '\n')
        f.write(str(tensor_variable.shape) + '\n')
        f.write(str(tensor_variable.numpy()) + '\n')

# 展示acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']

val_acc = history.history['val_sparse_categorical_accuracy']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()