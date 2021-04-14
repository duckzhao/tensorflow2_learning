# 使用class构建学习fashion_mnist数据集的神经网络模型---六步法

import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train/255., x_test/255.

class fashion_model(tf.keras.Model):
    def __init__(self):
        super(fashion_model, self).__init__()
        # 开始定义网络结构
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.d2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        y = self.flatten(inputs)
        y = self.d1(y)
        return self.d2(y)
model = fashion_model()

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)

model.summary()