'''
在使用model.fit训练数据集的过程中，acc上升，loss下降，这些变化过程model.fit实际上都记录下来了，
使用history=model.fit接收训练过程返回值，可以以如下方式得到loss和acc的变化曲线数值记录
acc = history.history['sparse_categorical_accuracy']  训练集准确率
val_acc = history.history['val_ sparse_categorical_accuracy'] 测试集准确率
loos = history.history['loss'] 训练集loss
val_loss = history.history['val_loss'] 测试集loss

p.s. history记录训练集的loss或者acc都是一个epochs记录一次，然后返回一个列表
测试集的loss或者acc可能是 根据validation_freq来的，若validation_freq=1就一个epoch记录一次
'''

# 1.
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os

# 2.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train/255., x_test/255.

# 3.
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# 4.
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 判断是否有历史模型，有的话就加载
checkpoint_path = './check_point/mnist.ckpt'
if os.path.exists(checkpoint_path+'.index'):
    print('-------------load checkpoint-------------')
    model.load_weights(checkpoint_path)

# 实例化保存模型的checkpoint对象
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=True)

# 5.
history = model.fit(x_train, y_train, 32, 5, validation_data=(x_test, y_test), validation_freq=1, callbacks=[cp_callback])

# 6.
model.summary()

# 设置打印不省略
np.set_printoptions(threshold=np.inf)
with open('./trainable_variables.txt',mode='w')as f:
    for tensor_variable in model.trainable_variables:
        print(tensor_variable.name)
        print(tensor_variable.shape)
        f.write(str(tensor_variable.numpy()) + '\n')


###############################################    show   ###############################################
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

print(acc)
print(len(acc))
print(val_acc)
print(len(val_acc))
print(loss)

plt.subplots_adjust(hspace=0.5)
# 指定下面语句操作的子图位置
plt.subplot(1, 2, 1)
# 在一个图里画出 训练集和测试集的准确率 两个折线图
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training adn Validation Accuracy')
plt.legend()

# 在一个图里画出 训练集和测试集的loss 两个折线图
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

# 展示整个大图
plt.show()