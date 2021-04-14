# 在InceptionNet中引入了block的概念，论文中常见这种block，实际上是一种面向对象的神经网络模型开发方式
# 把神经网络中重复的或比较相似的结构定义为一个block，方便调用和定义，节省工作量！大型重复网络常用
# 并且把 都使用 CBA 方式的卷积层也定义为一个类，这样每次只用一行代码（给CBA的卷积类传入必要参数）就可以一行实现卷积层了---需要写在Model类里（还是要有call）


# 开始写baseline代码，之后的cnn网络都可以从这个模板中加以修改，数据源此时都用cifar10
# 这种网络模型 训练时对于 输入图像 的维度没有要求，但是验证时输入图像的维度必须和训练的维度保持一致
# Conv2D在执行卷积的过程中是可以对三通道图片进行处理的，使用的filters也会是三通道的，（可能要通过input_shape说明下才行,不需要，自动根据输入改变kernal的深度）

# 1.
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, GlobalAveragePooling2D
import numpy as np
np.set_printoptions(threshold=np.inf)

# 2.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train/255., x_test/255.
print('训练集x的维度：', x_train.shape, '结果y的维度：', y_train.shape)

# 3.
# 由于inception块中的卷积都是CBA结构的，比较规范，因此将CBA卷积操作定义为一个类，便于调用卷积操作，一行即可
# 他在被调用时，就当成Conv2D一样用就行了，在init里面实例化写就ok，然后call里面再调用下返回y即可
class ConvBnAct(tf.keras.Model):
    # 在构造函数中声明一些必要参数
    def __init__(self, filter_num, kernel_size, strides=1, padding='same', activation='relu'):
        super(ConvBnAct, self).__init__()
        # 先采用sequential方式开发
        self.model = tf.keras.models.Sequential([
            Conv2D(filters=filter_num, kernel_size=kernel_size, strides=strides, padding=padding),
            BatchNormalization(),
            Activation(activation=activation)
        ])
        # 还可以自己写
        # self.conv1 = Conv2D(filters=filter_num, kernel_size=kernel_size, strides=strides, padding=padding)
        # self.bn1 = BatchNormalization()
        # self.act1 = Activation(activation=activation)

    def call(self, inputs, training=None, mask=None):
        y = self.model(inputs)
        # 或者
        # x = self.conv1(inputs)
        # x = self.bn1(x)
        # y = self.act1(x)
        return y


# 然后再定义出inception block结构的类方法， 在inceptionblock内，四组卷积的filters数量都相同，都以strides为1卷积，same padding
# 因此四组卷积输出的xy大小都是原图输入的xy大小，四组输出深度为filters，然后将四组输出叠加，最后inception块的输出结构为 (x,y,4*filters)数量的矩阵
class InceptionBlock(tf.keras.Model):
    # 在构造函数中声明每个block可能不同的必要参数---filter_num,strides
    def __init__(self, filter_num, strides):
        super(InceptionBlock, self).__init__()
        # 一共有四组卷积，每组卷积核的大小啥的都是固定的,激活函数都是relu所以不写了
        # 第一组
        self.conv1 = ConvBnAct(filter_num=filter_num, kernel_size=1, strides=strides)
        # 第二组
        self.conv2_1 = ConvBnAct(filter_num=filter_num, kernel_size=1, strides=strides)
        self.conv2_2 = ConvBnAct(filter_num, kernel_size=3, strides=1)  # 第二个卷积步长固定
        # 第三组
        self.conv3_1 = ConvBnAct(filter_num, kernel_size=1, strides=strides)
        self.conv3_2 = ConvBnAct(filter_num, kernel_size=5, strides=1)  # 第二个卷积步长固定
        # 第四组
        self.pool4 = MaxPool2D(pool_size=3, strides=1, padding='same')  # 池化步长固定为1，保证不减小层间传递的大小
        self.conv4 = ConvBnAct(filter_num, kernel_size=1, strides=strides)

    def call(self, inputs, training=None, mask=None):
        # 1
        x1 = self.conv1(inputs)

        # 2
        x2 = self.conv2_1(inputs)
        x2 = self.conv2_2(x2)

        # 3
        x3 = self.conv3_1(inputs)
        x3 = self.conv3_2(x3)

        # 4
        x4 = self.pool4(inputs)
        x4 = self.conv4(x4)

        # 将四组结果沿深度方向拼接在一起，axis=3方向
        y = tf.concat([x1, x2, x3, x4], axis=3)
        return y


# 开始实现inception网络，直接实例化inception block就行，快速开发
class InceptionNet(tf.keras.Model):
    # 在构造方法中定义block的数量， filters的数量（每个block输出深度）,class_num（最后分类全连接层数量）
    def __init__(self, block_num, filter_num, class_num):
        super(InceptionNet, self).__init__()
        # 当没有BN操作时时可以把激活直接写在卷积操作里
        # 1.一个3*3的卷积
        self.conv1 = ConvBnAct(filter_num=filter_num, kernel_size=3)
        # 2.一个大的block，里面包含两个inception block（第一个步长为2，第二个步长为1），第一个深度为filters
        # 3.一个大的block，里面包含两个inception block（第一个步长为2，第二个步长为1），第二个因为输入尺寸减小，加深深度，增强信息提取力,filters*2
        # 为了方便我们写一个sequential结构来把实例化的block放进去（因为block之间是顺序结构的），用一个循环实例化并放入block到sequential里去
        self.inception_blocks = tf.keras.models.Sequential()   # 使用add方法可以给里面加东西
        for block_id in range(block_num):
            for layer_id in range(2):
                # 如果是第一层inception block，步长为2
                if layer_id == 0:
                    block = InceptionBlock(filter_num=filter_num, strides=2)
                else:
                    block = InceptionBlock(filter_num=filter_num, strides=1)
                # 将小的 实例化的 InceptionBlock 加入 sequential结构中
                self.inception_blocks.add(block)
            # 当第一层大block实例化完了之后，给第二层大的block中的filters——num加深，增强信息承载力，第二层的输入来自第一层 4*filters的深度
            filter_num = filter_num*2
        # 4.
        self.avgpool = GlobalAveragePooling2D()
        # 5.
        self.dense1 = Dense(units=class_num, activation='softmax')

        '''
        当然也可以把大block结构拆开写，只不过就是重复写的多一点而已，都可以的，init里面的sequential结构，在call里面也是一个用法
        '''

    def call(self, inputs, training=None, mask=None):
        # print('input shape：', inputs.shape)  #  (None, 32, 32, 3) 图片是32 32大小的3通道数据
        x = self.conv1(inputs)
        # print('after conv1 shape：', x.shape)   # (None, 32, 32, 16)  因为卷积步长为1，所以输出xy大小一致，都是32，深度为filters数量
        x = self.inception_blocks(x)
        # print('after model shape：', x.shape)  # (None, 8, 8, 128) 因为两个大block都中的第一层都有strides为2卷积，所以输出尺寸是输入尺寸的一半，32/2=16/2=8，所以xy是8
        # 第一层大block的输出深度=16（filter_num）*4=64，（输出尺寸为16，16，64） 第二层大block的输出深度=32（filter_num）*4=128，（输出尺寸为8，8，128）---filter_num在第一层结尾*2了
        x = self.avgpool(x)
        y = self.dense1(x)
        return y

# cifar10  10分类，因此class_num=10，弄个小的block，只有两层
baseline_model = InceptionNet(filter_num=16, block_num=2, class_num=10)

# 4.
baseline_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                       metrics=['sparse_categorical_accuracy'])

# 5.
# 再fit之前设置保存model，并且判断能否加载历史训练参数
check_point_path = './check_point/InceptionNet.ckpt'
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