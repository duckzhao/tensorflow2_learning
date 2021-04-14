# 当数据集比较少或者实际给出的数据集和案例数据集可能不完全一致时（方向偏移、旋转、宽度偏移）
# 对图像的增强就是对图像进行简单形变，解决因为拍照角度不同等因素造成的影响
'''可以使用图像增强技术，api如下：tf.keras.preprocessing.image.ImageDataGenerator(
    rescale =对输入的所有特征都乘以该数值，当输入 1/255.时，可对图像像素进行0-1的归一化
	rotation_ range =随机旋转角度数范围， 45
	width_ shift range =随机宽度偏移量， float,指宽度偏移的比例
	height shift range =随机高度偏移量，
	水平翻转: horizontal_flip =是否随机水平翻转， 实际上是左右镜像图片，False
	随机缩放: zoom_range =随机缩放的范围[1-n, 1+n] ，随机缩小或者放大图片的比例,
	浮点数 或 [lower, upper]。随机缩放范围。如果是浮点数，[lower, upper] = [1-zoom_range, 1+zoom_range])   '''
# 1.
import tensorflow as tf

# 2.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# print(x_train.shape)
x_train, x_test = x_train/255., x_test/255.

# 由于ImageDataGenerator实例化后在fit时直接收 四维数据，因此需要把x_train转换为四维，单通道下，实际上就是多了一个[]包裹，其余的没啥变化
x_train = x_train.reshape(len(x_train), 28, 28, 1)   # 或者用x_train.shape[0]

# 实例化一个ImageDataGenerator对象，并定义对图片增强的具体参数
img_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale= 1./1., # 不放大或缩小数值元素
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=False,  # 数字和衣服什么的一般也不会出现反转情况的源数据
    zoom_range=0.5  # 则范围是[0.5, 1.5]
)
# img_gen_train.fit指定要增强的数据集x_train,输入参数必须是四维数组，即使只有一个通道
img_gen_train.fit(x_train)

# 3.
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

#4.
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 5.
# 当使用ImageDataGenerator数据增强后，model.fit中传入的前三个参数x，y，batch_size，从img_gen_train.flow中传入
model.fit(img_gen_train.flow(x=x_train, y=y_train, batch_size=32), epochs=5, validation_data=(x_test, y_test), validation_freq=1)

#6.
model.summary()