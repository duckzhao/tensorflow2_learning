'''
使用简单的模型和vggnet训练好的cat和dog分类模型，完成预测
'''

from cats_vs_dogs_classify_vgg_model import _decode_and_resize
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Dense, Flatten
import tensorflow as tf
import numpy as np
import os

data_dir = r'./fastai-datasets-cats-vs-dogs-2'
# model_path = data_dir + '/models/VGGNet_cat_dog.ckpt'
model_path = data_dir + '/vgg_models/VGGNet_cat_dog.ckpt'


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

    # 训练model的时候没加 @tf.function ，调用model时加了  @tf.function ，依旧可以成功调用model
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

simple_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 5, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model = VGGNet()
# model = simple_model

model.load_weights(filepath=model_path)

resized_pic, _ = _decode_and_resize(filename=r'C:\Users\Z\Desktop\fastai-datasets-cats-vs-dogs-2\test1\zk.jpg', label=1)
# print(resized_pic.shape)    (256, 256, 3)
# 给图片新增一个维度
resized_pic = resized_pic[tf.newaxis, ]
# print(resized_pic.shape)    # (1, 256, 256, 3)
label = model.predict(resized_pic).argmax()
print('预测的结果是:', 'dog' if label else 'cat')



'''
使用evaluate完成对模型在测试集上的评估，simple_model仅有63左右的测试准确率,vggnet准确率高达95
'''

# --------------------------------------开始测试------------------------------------------- #
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
test_dataset = test_dataset.batch(64)

# 在测试之前需要将model进行compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

print(model.metrics_names)
print(model.evaluate(x=test_dataset))
