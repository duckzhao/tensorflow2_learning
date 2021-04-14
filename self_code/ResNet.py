'''
单纯堆叠神经网络的深度会造成退化现象，即后面的特征丢失了前边特征原始的模样。
resnet提出了层间残差跳连，使用一跟跳连线，把前面的特征在不经处理的情况下，送到后面几层的网络上，直接和后面几层网络矩阵做加法运算
这一操作缓解了由于神经网络层数太多造成的退化现象。
跳连分为两种情况①跳连前后矩阵维度相同，直接加法运算即可；②跳连前后矩阵维度不同，前面的矩阵xy大，后面的矩阵xy小（深度不一定），
这种情况下 对前面的大矩阵采用1*1的卷积核进行调整，调整使得前面的矩阵和后面的矩阵维度一致，然后再做加法。
1*1的卷积核通过strides改变特征图的尺寸，通过filters改变特征图的深度，大多数都是对前面调过来的矩阵操作的
p.s. 一般只有当跳接中间有strides为2 ，或者filters和前面深度不一样，，，的操作时才需要调整，strides为1不改变xy大小，深度可能改变了
'''

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
x_train, x_test = x_train / 255., x_test / 255.


# 3.
# resnet block类，其任务是，对于一个输入x1，计算完成对x1卷积(两层)后的结果x2，然后输出x2+x1=y（可能要根据x2的维度对x1进行微调）
# 参数有①卷积操作中filters数量；②卷积的步长；③是否需要调整跳连的x1大小 residual_path = 1调整/0不调整
# 按理说卷积核的大小也应该传进来，但是似乎resnet里卷积核固定是3*3，所以就不传了
class ResBlock(tf.keras.Model):
    def __init__(self, filter_num, strides, residual_path):
        super(ResBlock, self).__init__()
        self.residual_path = residual_path  # 方便其他函数能够调用这个值
        # 1.完成两层卷积
        # 第一层卷积核步长大小是输入的这个strides，不使用bias,CBA
        self.conv1 = Conv2D(filters=filter_num, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        # 第二层步长固定为1不使用bias，CBA，A在跳连求和以后用
        self.conv2 = Conv2D(filters=filter_num, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = BatchNormalization()

        # 根据 residual_path 判断是否要调整inputs的大小,调整称为下采样
        # 如果真就调整---即filters不等于输入特征的深度或者strides ！= 1
        if residual_path:
            # filter_num使得深度保持一致，strides使得大小保持一致------重复第一层卷积的strides即可（可能使得原来矩阵的特征跳跃，没有完全保留晚来矩阵的特征）
            self.down_c1 = Conv2D(filters=filter_num, strides=strides, kernel_size=1, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.act2 = Activation('relu')  # 这个激活是跳连相加后一起激活

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # 接下来要么是 x直接加inputs，要么是x+input调整后的大小，得到y
        # 如果真，则使用下采样保持维度一致
        if self.residual_path:
            inputs = self.down_c1(inputs)
            inputs = self.down_b1(inputs)
        # 否则说明维度一致，直接加就好了
        y = self.act2(x + inputs)
        return y


class ResNet(tf.keras.Model):
    # block_size-list,长度表示大的resnet block的数量，一个大的resnet block里面有n个ResBlock，具体数量用list中的数字表示
    # 【2，2，2，2】表示有4个大的resnet block，且每个大的block中都有两个小的ResBlock，（即4个卷积层，两个跳连相加操作）
    # initial_filters表示初始的卷积深度，每经过一个大的block，filters深度*2
    def __init__(self, block_size, initial_filters=64):
        super(ResNet, self).__init__()
        # 1.
        self.conv1 = Conv2D(filters=initial_filters, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')

        # 2.接下来是4个连续的大block，放到一个sequential结构里，这样就不用在call里多写几行了
        self.blocks = tf.keras.models.Sequential()
        # 遍历4个大的block
        for block_index in range(len(block_size)):
            # 遍历每个block中的小block
            for layer_index in range(block_size[block_index]):
                # 如果是第一层的block_index，则是两层直连小block
                if block_index == 0:
                    block = ResBlock(filter_num=initial_filters, strides=1, residual_path=False)
                # 如果是其他的大block
                else:
                    # 判断layer_index是不是0，0的层是需要向下采样调整原始矩阵大小的，residual_path=True
                    # 具体原因是 strides为2，每层大的block中filters数量都是一样的
                    if layer_index == 0:
                        block = ResBlock(filter_num=initial_filters, strides=2, residual_path=True)
                    else:
                        block = ResBlock(filter_num=initial_filters, strides=1, residual_path=False)
                # 每一小循环都加入一个block
                self.blocks.add(block)
            # 每一个大循环改变一次filters num
            initial_filters *= 2

        # 3.
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        # 4.
        self.dense = Dense(units=10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        y = self.dense(x)
        return y


baseline_model = ResNet([2, 2, 2, 2], initial_filters=64)

# 4.
baseline_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                       metrics=['sparse_categorical_accuracy'])

# 5.
# 再fit之前设置保存model，并且判断能否加载历史训练参数
check_point_path = './check_point/ResNet.ckpt'
if os.path.exists(check_point_path + '.index'):
    print('-------------load the mode1-------------')
    baseline_model.load_weights(check_point_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_point_path, save_weights_only=True, save_best_only=True)

history = baseline_model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test),
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
