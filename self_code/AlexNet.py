# 开始写baseline代码，之后的cnn网络都可以从这个模板中加以修改，数据源此时都用cifar10
# 这种网络模型 训练时对于 输入图像 的维度没有要求，但是验证时输入图像的维度必须和训练的维度保持一致
# Conv2D在执行卷积的过程中是可以对三通道图片进行处理的，使用的filters也会是三通道的，（可能要通过input_shape说明下才行,不需要，自动根据输入改变kernal的深度）

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
class AlexNet(tf.keras.Model):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 当没有BN操作时时可以把激活直接写在卷积操作里 #
        self.conv1 = Conv2D(filters=96, kernel_size=(3, 3), strides=1)
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        self.pool1 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.conv2 = Conv2D(filters=256, kernel_size=3, strides=1)
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')
        self.pool2 = MaxPool2D(pool_size=3, strides=2)

        self.conv3 = Conv2D(filters=384, kernel_size=3, padding='same', activation='relu')
        self.conv4 = Conv2D(filters=384, kernel_size=3, padding='same', activation='relu')
        self.conv5 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')
        self.pool5 = MaxPool2D(pool_size=3, strides=2)

        self.flatten = Flatten()
        self.dense1 = Dense(units=2048, activation='relu')
        self.drop1 = Dropout(0.5)
        self.dense2 = Dense(units=2048, activation='relu')
        self.drop2 = Dropout(0.5)
        self.dense3 = Dense(units=10, activation='softmax')


    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool5(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.drop1(x)
        x = self.dense2(x)
        x = self.drop2(x)
        y = self.dense3(x)
        return y

baseline_model = AlexNet()

# 4.
baseline_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                       metrics=['sparse_categorical_accuracy'])

# 5.
# 再fit之前设置保存model，并且判断能否加载历史训练参数
check_point_path = './check_point/AlexNet.ckpt'
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