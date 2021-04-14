# 将模型训练结果中的  被训练参数名称，shape，变量值  都打印出来，并存储到本地
# 在上一讲断点续训的代码基础上加以改进
'''
model.trainable_variables  该模型属性返回模型中可训练的参数tensor列表，每个tensor的属性有其训练结果，大小，名称等等，可以直接遍历打印和存储！
好像，model必须fit之后才能有这个trainable_variables属性，就算给model加载历史训练参数，调用trainable_variables也会报错（或者还有其他等效fit的手段）

由于我们一般打印时 如果待打印内容非常多，pycharm就会省略打印内容，使用如下语句可以不省略打印
np.set printoptions(threshold=超过多少省略显示)
np.set_printoptions(threshold=np.inf) # np. inf表示无限大，即不省略
'''


# 1.
import tensorflow as tf
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

#4.
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 新增步骤判断是否已有历史模型，有的话就加载
# 可以指定一个不存在的空文件夹存模型，后续写入的时候，会自动创建这个空文件夹，不会报错
check_point_path = './check_point/mnist.ckpt'
# 一般如果有历史ckpt模型参数，也会有一个index文件的，用index文件判断是否存储过模型
if os.path.exists(check_point_path+'.index'):
    print('-------------load the mode1-------------')
    model.load_weights(check_point_path)

# 新增步骤，配置控制存储模型的 ModelCheckpoint对象
# 使用当前配置，如果模型存储路径不变，则只会保留一个模型参数文件---保留最优的？还是最新的？,filepath就是 xxxx.ckpt地址
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_point_path, save_best_only=True, save_weights_only=True)

# 5.
history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])

# 6.
model.summary()

#打印不省略的模型参数并存储到txt文本中
import numpy as np
np.set_printoptions(threshold=np.inf)
print(model.trainable_variables)    # 这样打印出来的就是不省略的了
# 遍历存储trainable_variables
with open(r'./trainable_variables.txt', mode='w')as f:
    # model.trainable_variables返回的可迭代对象是可训练的tensor变量对象，几个tensor变量几个返回迭代对象
    # 每个变量有多个属性name，shape，numpy()可以把tensor对象转为numpy格式
    for tensor_variable in model.trainable_variables:
        f.write(str(tensor_variable.name) + '\n')
        f.write(str(tensor_variable.shape) + '\n')
        f.write(str(tensor_variable.numpy()) + '\n')
        print(str(tensor_variable.name), tensor_variable.shape)