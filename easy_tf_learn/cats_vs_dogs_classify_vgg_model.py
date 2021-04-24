'''
以下代码以猫狗图片二分类任务为示例，展示了使用 tf.data 结合 tf.io 和 tf.image 建立 tf.data.Dataset 数据集，并进行训练和测试的完整过程。
训练和测试分开进行，训练使用fit完成，测试使用evaluate完成

* 如果我们使用的是自定义class model，而非sequential结构model，务必在 map 函数中 加入这一行  image_resized.set_shape([256, 256, 3])
以指定该图片的维度，否则报 The channel dimension of the inputs should be define 错误。图片通道数没有被指定
* 使用sequential结构作为 model 时，给 map 函数中加上这句话也ok的。

'''
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Dense, Flatten
import os
import numpy as np

# data_dir = r'C:\Users\Z\Desktop\fastai-datasets-cats-vs-dogs-2'
data_dir = r'./fastai-datasets-cats-vs-dogs-2'

# 当网络结构比较复杂，且数据集样本像素比较大时，不应该设置过大的batch_size，gpu显存在完成神经网络计算时会不足，程序运行不起来
batch_size = 32
learning_rate = 0.001
num_epochs = 10


# 定义应用于map的函数，该函数首先完成从图片地址读取图片的功能，其次统一图片大小并完成归一化操作
# p.s. 本数据集中图片格式都是jpg，但图片大小不同，统一resize为[256, 256]
def _decode_and_resize(filename, label):
    '''
    :param filename: 该函数应用于map时，会从dataset对象中单个送入filename地址
    :param label: 以及与该地址图片匹配的label
    :return: 处理后的真实图片数据，[256, 256]
    '''
    image_string = tf.io.read_file(filename=filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    # 完成resize，顺便归一化了
    image_resized = tf.image.resize(images=image_decoded, size=[256, 256]) / 255.
    # 使用class类型的model时，这一句必须要加，虽然加了前后image_resized的shape没变化，但是就是要加，否则报不能找到图片channel通道错误
    image_resized.set_shape([256, 256, 3])
    return image_resized, label


class VGGNet(tf.keras.Model):
    def __init__(self):
        super(VGGNet, self).__init__()
        # 当没有BN操作时时可以把激活直接写在卷积操作里 #
        self.conv1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.bn1 = BatchNormalization()
        self.act1 = Activation(activation='relu')

        self.conv2 = Conv2D(filters=64, kernel_size=3, padding='same')
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')
        self.pool2 = MaxPool2D(2, 2, padding='same')
        self.drop2 = Dropout(0.2)

        self.conv3 = Conv2D(filters=128, kernel_size=3, padding='same')
        self.bn3 = BatchNormalization()
        self.act3 = Activation('relu')

        self.conv4 = Conv2D(filters=128, kernel_size=3, padding='same')
        self.bn4 = BatchNormalization()
        self.act4 = Activation('relu')
        self.pool4 = MaxPool2D(pool_size=2, strides=2, padding='same')
        self.drop4 = Dropout(0.2)

        self.conv5 = Conv2D(filters=256, kernel_size=3, padding='same')
        self.bn5 = BatchNormalization()
        self.act5 = Activation('relu')

        self.conv6 = Conv2D(filters=256, kernel_size=3, padding='same')
        self.bn6 = BatchNormalization()
        self.act6 = Activation('relu')

        self.conv7 = Conv2D(filters=256, kernel_size=3, padding='same')
        self.bn7 = BatchNormalization()
        self.act7 = Activation('relu')
        self.pool7 = MaxPool2D(pool_size=2, strides=2, padding='same')
        self.drop7 = Dropout(0.2)

        self.conv8 = Conv2D(filters=512, kernel_size=3, padding='same')
        self.bn8 = BatchNormalization()
        self.act8 = Activation('relu')

        self.conv9 = Conv2D(filters=512, kernel_size=3, padding='same')
        self.bn9 = BatchNormalization()
        self.act9 = Activation('relu')

        self.conv10 = Conv2D(filters=512, kernel_size=3, padding='same')
        self.bn10 = BatchNormalization()
        self.act10 = Activation('relu')
        self.pool10 = MaxPool2D(pool_size=2, strides=2, padding='same')
        self.drop10 = Dropout(0.2)

        self.conv11 = Conv2D(filters=512, kernel_size=3, padding='same')
        self.bn11 = BatchNormalization()
        self.act11 = Activation('relu')

        self.conv12 = Conv2D(filters=512, kernel_size=3, padding='same')
        self.bn12 = BatchNormalization()
        self.act12 = Activation('relu')

        self.conv13 = Conv2D(filters=512, kernel_size=3, padding='same')
        self.bn13 = BatchNormalization()
        self.act13 = Activation('relu')
        self.pool13 = MaxPool2D(pool_size=2, strides=2, padding='same')
        self.drop13 = Dropout(0.2)

        self.flatten = Flatten()
        self.dense14 = Dense(units=512, activation='relu')
        self.drop14 = Dropout(0.2)
        self.dense15 = Dense(units=512, activation='relu')
        self.drop15 = Dropout(0.2)
        self.dense16 = Dense(units=2, activation='softmax')

    # 不开启tf.function的model，可以接着开启tf.function继续训练。开启后的训练速度确实加快了，但会出现 WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time
    # 大概意思是，准备batch的时间，比训练batch的时间都长？好像和callback有点关系，看不太明白反正。不过应该不要紧，这个warning
    # 实践证明linux下加上 @tf.function似乎也不能开启tensorboard中的garph，且有时会造成莫名其妙的bug，训练集训练准确率上不去
    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.pool4(x)
        x = self.drop4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.act6(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.act7(x)
        x = self.pool7(x)
        x = self.drop7(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.act8(x)

        x = self.conv9(x)
        x = self.bn9(x)
        x = self.act9(x)

        x = self.conv10(x)
        x = self.bn10(x)
        x = self.act10(x)
        x = self.pool10(x)
        x = self.drop10(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = self.act11(x)

        x = self.conv12(x)
        x = self.bn12(x)
        x = self.act12(x)

        x = self.conv13(x)
        x = self.bn13(x)
        x = self.act13(x)
        x = self.pool13(x)
        x = self.drop13(x)

        x = self.flatten(x)
        x = self.dense14(x)
        x = self.dense15(x)
        y = self.dense16(x)

        return y


if __name__ == '__main__':
    # 构建训练数据集->生成图片地址和对应标签的Dataset对象
    # 1.生成训练集图片路径tensor并合并训练集
    train_cat_filenames = tf.constant(value=[os.path.join(os.path.join(data_dir, 'train/cats'), path) for path in
                                             os.listdir(os.path.join(data_dir, 'train/cats'))])
    # print(train_cat_filenames.shape)    # (11500,)
    train_dog_filenames = tf.constant(value=[os.path.join(os.path.join(data_dir, 'train/dogs'), path) for path in
                                             os.listdir(os.path.join(data_dir, 'train/dogs'))])
    # print(train_dog_filenames.shape)    # (11500,)
    train_filenames = tf.concat([train_cat_filenames, train_dog_filenames], axis=0)
    # print(train_filenames.shape)    # (23000,)
    # 构建数据集标签，0->cat 1->dog
    train_labels = tf.concat([tf.zeros(shape=train_cat_filenames.shape), tf.ones(shape=train_dog_filenames.shape)], axis=0)
    # print(train_labels)  # (23000,)   全都是行向量，所以axis都可以填 -1，指定横向拼接

    # 提前打乱训练集地址顺序和labels顺序
    random_index = np.random.permutation(train_filenames.shape[0])
    train_filenames = train_filenames.numpy()[random_index]
    train_labels = train_labels.numpy()[random_index]

    # 2.将train特征(图片路径，并没有加载图片到内存，否则低效率)和labels生成Dataset对象的配对数据集
    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))

    # 3.对dataset数据集进行预处理->从filepath到图片tensor，及批处理操作（分批、resize、归一化）等
    train_dataset = train_dataset.map(map_func=_decode_and_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)    # 开启map操作的cpu并行加速
    # 因为之前已经手动打乱过了，所以这里的buffersize不应该设置过大，否则会有一些内存溢出的问题
    train_dataset = train_dataset.shuffle(buffer_size=2000)
    train_dataset = train_dataset.batch(batch_size=batch_size)
    # 开启训练数据的预加载功能，cpu加载训练数据与gpu训练过程并行
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


    # 加载测试集
    test_cat_filenames = tf.constant(value=[os.path.join(os.path.join(data_dir, 'valid/cats'), path) for path in
                                             os.listdir(os.path.join(data_dir, 'valid/cats'))])
    test_dog_filenames = tf.constant(value=[os.path.join(os.path.join(data_dir, 'valid/dogs'), path) for path in
                                             os.listdir(os.path.join(data_dir, 'valid/dogs'))])
    test_filenames = tf.concat([test_cat_filenames, test_dog_filenames], axis=0)
    test_labels = tf.concat([tf.zeros(shape=test_cat_filenames.shape), tf.ones(shape=test_dog_filenames.shape)], axis=0)

    # 构造dataset测试集
    test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
    test_dataset = test_dataset.map(_decode_and_resize)
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # 使用vggnet进行训练
    model = VGGNet()

    # metrics的设置参数属于训练优化 待训练矩阵 的参数，种类也比较多，不过和测试集没啥关系，就是交叉熵损失，还是准确率等等指标之间的切换
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])

    # 再fit之前设置保存model，并且判断能否加载历史训练参数
    check_point_path = './fastai-datasets-cats-vs-dogs-2/vgg_models/VGGNet_cat_dog.ckpt'
    if os.path.exists(check_point_path + '.index'):
        print('-------------load the mode1-------------')
        model.load_weights(check_point_path)

    # 如果我们fit时没有指定测试集，则无法得到val_loss的数据，就没法根据这个monitor选择保存最佳的model，所以就不会保存model，因此没有
    # 测试集时，需要修改默认的monitor为'loss'，以训练集损失最小为指标保存最优模型
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_point_path, save_weights_only=True,
                                                     save_best_only=True, monitor='val_loss')
    # 保存logs日志，用于tensorboard可视化
    tensorboard_log_path = './vgg_cat_dog_logs/'
    # 如果该地址不存在就创建一个logs文件夹
    if not os.path.exists(tensorboard_log_path):
        os.mkdir(tensorboard_log_path)
    # 可以每个batch保存一次loss或者别的信息，指定 updata_freq='batch'即可
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_path, histogram_freq=1, update_freq='batch')

    # 使用evaluate方式验证model效果，不如这种方式好，该可以时时观察是否过拟合，以及可以直接在tensorboard中监控到val_loss等值
    model.fit(x=train_dataset, epochs=num_epochs, callbacks=[cp_callback, tensorboard_callback],
              validation_data=test_dataset, validation_freq=1)