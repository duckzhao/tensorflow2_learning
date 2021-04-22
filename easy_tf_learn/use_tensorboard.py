'''
大家还是要知道一些额外的事情的，就是在使用自定义class方式构建网络model时，这个自定义class他也是符合python面向对象基本原则的
其中init函数里是初始化一些layers，只在类实例化时执行一次，是可以在里面加一些print或者一些验证的计算的，用于查看该类在实例化时的一些状态

其中call方法，隐式的被__call__方法所调用，call方法传出去的“类实例化对象调用call”，在model.fit时，call方法中的print并不会一直随着训练的
batch_size一直被循环调用，只会在开始fit读入model模型内容时，调用两次，打印一些相关信息，此后就不会再调用了，
可能是因为tf的api没有录入print这句话到model训练过程去，因为他不识别print是和tf有关的api所以不录入？？？
'''

'''
模型的训练：如果按照北大的tf2教程来看的话，我们训练模型直接在model.fit中就训练了，过于简单化，实际上我们也可以自己写一个epoch循环，写一个load 指定
batch_size data的数据加载器，然后实例化单独的loss函数和optimizer完成自定义的训练，这样可以加入更多的自定义的一些东西。

模型的测评：在北大教程里我们在fit时就指定了validation_data了，有时当我们已经有训练好的模型，想拿到陌生的数据集上验证效果时，就不能用model.fit
方式了，此时可以使用model.evaluate(data_loader.test_data, data_loader.test_label)进行模型测评，即先构造model的网络框架，
然后 load 这个model，最后用这个model完成效果验证。（数据加载器解决了训练数据内存占用过大的问题）


首先简单粗暴的tf中的教了如何自定义属于自己的模型，即新建一个类，继承tf.keras.Model,然后在这个类里面可以写很多的自定义代码，改变顺序层，新建复杂跳跃连接层
如果是顺序结构，创新性不大，大多数都是利用的tf.keras.layers中的api，可以使用tf.keras.models.sequential([])结构新建模型

如果这两者都无法满足我们自己的需求，我们要新建特殊的、新的layer或者特殊的、新的（损失函数）求导方式，
新建特殊的layer时可以试试新建一个类，继承tf.keras.layers.Layer,新建属于，并重写 __init__ 、 build 和 call 三个方法，
自己的tf layers层，在class或者sequential方式调用的时候，就和调用tf.keras.layers.xxx中的层一样实例化然后调用就可以了，也可以指定关键构造参数

自定义新的损失函数的时候仅需要新建一个类继承tf.keras.losses.Loss类即可，构造方法 输入真实值 y_true 和模型预测值 y_pred，
输出模型预测值和真实值之间通过自定义的损失函数计算出的损失值。
https://tf.wiki/zh_hans/basic/models.html#zh-hans-custom-layer
'''

'''
在本节向大家展示如何使用tensorflow在使用model.fit模式进行训练时如何完成tensorboard的开启。--->利用fit参数callbacks完成
https://blog.csdn.net/wss794/article/details/105207494/
tf.keras.callbacks.TensorBoard利用该api即可，如果想在远程电脑服务器上开启tensorboard服务，然后在本地电脑上看，需要使用mobaxterm建立映射隧道
https://blog.csdn.net/axept/article/details/115016329 然后cd到tensorboard日志文件夹保存的上层目录，
terminal输入tensorboard --logdir="logs"即可   "logs"指向保存tensorboard logs的文件夹名称。似乎linux上保存的logs，
无法查看graphs结果，其余都可以正常查看，
linux上保存的logs文件夹可以拖到windows上使用tensorboard查看

如果使用更加自定义的logs保存模式，可以参考 https://tf.wiki/zh_hans/basic/tools.html#tensorboard这种方式
'''

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
        print('call中的自定义打印任务，打印训练中间变量：', x.shape)   # 只会打印2次
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

# 新增设置tensorboard观察训练过程中的参数变化情况
tensorboard_log_path = './logs/'
# 如果该地址不存在就创建一个logs文件夹
if not os.path.exists(tensorboard_log_path):
    os.mkdir(tensorboard_log_path)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_path, histogram_freq=1)


history = baseline_model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test),
                             validation_freq=1, callbacks=[cp_callback, tensorboard_callback])

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
        # f.write(str(tensor_variable.numpy()) + '\n')

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
