'''
介绍卷积神经网络层的搭建方法，即参数含义，并使用class形式的model完成标准卷积神经网络的训练模板搭建，
后续仅用修改class中的init和call函数即可实现其他cnn网络

p.s. 卷积神经网络就是特征提取器，C卷积 B批量标准化 A激活 P池化 D舍弃
tf.keras.layers.Conv2D (
	filters =卷积核个数， 即观察者的个数，
	kernel_size =卷积核尺寸，# 正方形写核长整数，或(核高h,核宽w)
	strides =滑动步长，# 横纵向相同写步长整数，或纵向步长h，横向步长w)， 默认1
	padding = "same" or "Valid", # 使用全零填充是"same",不使用是"valid" (默认)
	activation =“relu”or“sigmoid”or“tanh”or“softmax"等，# 如有BN此处不写
	input_shape = (高，宽，通道数)  # 输入特征图维度，可省略
)
# 输入特征图的深度（channel数）决定了卷积核的深度，当前层的卷积核的个数，决定了当前层输出特征图的深度
比如输入是 None*32*(20*20)的矩阵经过6个观察者filters进行卷积操作，传给下一层的结构就是 (None, 16, 16, 6)

这里提供一个Sequential示例：使用一行写出来卷积层，推荐使用23行的形式，带参数名的传入更加清晰可见，使用5行cbapd写一个卷积
model = tf.keras.mode.Sequential([
	Conv2D(6， 5，padding='valid', activation='sigmoid'),
	MaxPoo12D(2，2)，
	Conv2D(6, (5, 5)，padding=' valid', activation= 'sigmoid'),
	MaxPool2D(2, (2, 2))，
	Conv2D(filters=6， kerne1_size=(5，5), padding='valid', activation= 'sigmoid' )，
	MaxPool2D(pooI_s1ze=(2, 2), strides=2),
	Flatten(),
	Dense(10, activation='softmax')
])
分步Sequential示例：
model = tf.keras.models.Sequential([
	Conv2D(filters=6，kernel size=(5, 5)，padding='same')，#卷积层
	BatchNormalization()，# BN层位于卷积层之后，激活层之前。TF描述批标准化，如果不需要ban操作，可以把激活和卷积都写在Conv2D里面，如果需要BN操作，则要分三行写
	Activation('relu), # 激活层
	MaxPoo12D(poo1_size=(2, 2)，strides=2，padding='same'), # 池化层 AveragePooling2D使背脊更加平滑，MaxPoo12D更加注重背景的纹理（突出值）特征
	Dropout(0.2)，# dropout层，0.2表示随机舍弃掉20%的神经元
])
分步class示例：
self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding='same')  # 卷积层
self.b1 = BatchNormalization()  # BN层
self.a1 = Activation('relu')  # 激活层
self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
self.d1 = Dropout(0.2)  # dropout层

self.flatten = Flatten()
self.f1 = Dense(128, activation='relu')
self.d2 = Dropout(0.2)
self.f2 = Dense(10, activation='softmax')
'''


# 开始写baseline代码，之后的cnn网络都可以从这个模板中加以修改

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

# 3.
class Baseline(tf.keras.Model):
    def __init__(self):
        super(Baseline, self).__init__()
        # 第一层就是卷积层， 卷积计算就是特征提取器，CBAPD
        self.conv1 = Conv2D(filters=6, kernel_size=(5, 5), strides=1, padding='same')  # 卷积层
        self.bn1 = BatchNormalization()  # 不用写参数，但是会多计算几个变量，标准化卷积完了输出至正态分布时的变量,一个卷积核多2个变量
        self.act1 = Activation(activation='relu')  # 激活层
        self.pool1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.drop1 = Dropout(0.2)  # dropout层

        self.flatten = Flatten()
        # 第二层
        self.dense1 = Dense(units=128, activation='relu')
        self.drop2 = Dropout(0.2)
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)  # 一个filter出来一个32，32的xy矩阵，（三通道的卷积核卷积完之后，是当作一维变量求和的，求和结果是一个数字（并非三维），所以卷积核的数量决定了卷积层输出的深度）
        # print(x.shape)  # 确认了卷积后输出的参数矩阵的深度是6 (None, 32, 32, 6) 卷积过程中，卷积核永远只有5*5*3*6个 训练参数
        # 卷积后前向传播矩阵中的都是计算数值，不算训练参数，它的大小当使用same填充，且步长为1时，是和输入矩阵等大小的，而且深度还变成了6（32，32，6）
        # 这个大小后面去直接展开全连接层传入是很恐怖的，因此需要池化下，减小传播的计算矩阵维度
        x = self.bn1(x)
        # print(x.shape)  # (None, 32, 32, 6)   # 批量标准化不改变矩阵大小
        x = self.act1(x)
        # print(x.shape)  # (None, 32, 32, 6)   # 激活不改变矩阵大小
        x = self.pool1(x)
        print(x.shape)  # (None, 16, 16, 6)  # 池化改变了向后传播的矩阵大小，缩小了全连接层前的矩阵大小，减少全连接层参数
        x = self.drop1(x)
        # print(x.shape)    # (None, 16, 16, 6)   # 舍弃不改变矩阵大小
        x = self.flatten(x)  # flatten里面也要穿参数！！！
        # print(x.shape)  # (None, 1536)    1536 = 16*16*6
        x = self.dense1(x)
        x = self.drop2(x)
        y = self.dense2(x)
        return y

baseline_model = Baseline()

# 4.
baseline_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                       metrics=['sparse_categorical_accuracy'])

# 5.
# 再fit之前设置保存model，并且判断能否加载历史训练参数
check_point_path = './check_point/cifar10.ckpt'
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