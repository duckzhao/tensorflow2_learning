# 使用sequential构建学习fashion_mnist数据集的神经网络模型---六步法

import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train/255., x_test/255.

model = tf.keras.models.Sequential([
    # 输入也是28*28的像素点
    tf.keras.layers.Flatten(),
    # 模拟mnist的两层网络
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x=x_train, y=y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)

model.summary()