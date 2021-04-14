# 把训练的模型参数保存到本地，首先可以防止训练到一半意外停止，以前的训练结果都丢失
# 其次，把训练的最优模型参数保存，下次可以直接加载这个模型，然后去做对新的输入的预测
# 注意！这样保存的模型，没有办法从上次训练断掉的的数据集上接着训练，只能接着训练参数优化，数据集是从头开始再训练的
'''
1.定义存放模型的路径和文件名，命名为ckpt文件
2.生成ckpt文件时会同步生成index索引表，所以判断索引表是否存在，来判断是否存在模型参数
3.如有索引表，则直接读取ckpt文件中的模型参数
'''
'''
model.load_weights(ckpt_path)    使用本行代码给指定model加载历史模型参数
首先实例化一个 配置模型参数如何存储 的对象
cp_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath=路径文件名，
	save_weights_only= True/False,  # 是否只保留模型参数，一般True
	save_best_only=True/False)  # 是否只保留最优结果，一般True
然后将该对象填入model.fit的回调函数中，并以history接收
history = model.fit ( callbacks=[cp_callback] )
其余都和之前的神经网络模型编写一致
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
# 使用当前配置，如果模型存储路径不变，则只会保留一个模型参数文件---保留最优的？还是最新的？
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_point_path, save_best_only=True, save_weights_only=True)

# 5.
history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])

# 6.
model.summary()