# 利用已有的标注过的图片，自制mnist数据集，并使用sequential方式进行分类
# 定义generate函数，完成对标注图片的内存加载，numpy格式转换，并将内存中数据集存储为npy格式，以便下次直接调用

# 1.
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 2.导入数据集
# 图片文件夹路径
mnist_test_jpg_path = './MNIST_FC/mnist_image_label/mnist_test_jpg_10000/'
mnist_train_jpg_path = './MNIST_FC/mnist_image_label/mnist_train_jpg_60000/'
# 图片标签路径
mnist_test_label_path = './MNIST_FC/mnist_image_label/mnist_test_jpg_10000.txt'
mnist_tarin_label_path = './MNIST_FC/mnist_image_label/mnist_train_jpg_60000.txt'
# 数据集存储路径
mnist_x_train_npy = './MNIST_FC/mnist_image_label/mnist_x_train.npy'
mnist_y_train_npy = './MNIST_FC/mnist_image_label/mnist_y_train.npy'
mnist_x_test_npy = './MNIST_FC/mnist_image_label/mnist_x_test.npy'
mnist_y_test_npy = './MNIST_FC/mnist_image_label/mnist_y_test.npy'

# 传入图片路径和标签路径，返回匹配、读取好的 x，y_内存变量
def generate_dbs(jpg_path, label_path):
    with open(label_path, mode='r') as f:
        lines = f.readlines()
        lines = [__.replace('\n', '') for __ in lines]
    # print(lines[:10])
    # 存储数据集的两个空列表，后续转nparray格式
    x = []
    y_ = []
    for line in lines:
        pic_path = line.split()[0]
        pic_label = line.split()[1]
        pic_path = os.path.join(jpg_path, pic_path)
        # print(pic_path)
        # print(pic_label)
        pic = Image.open(pic_path)
        pic = np.array(pic.convert('L'))    # 转为灰度图片，然后转为28*28的二维矩阵
        # print(pic, pic.shape)
        pic = pic/255.  # 归一化
        x.append(pic)
        y_.append(pic_label)
        print('loading : ' + line)  # 打印状态提示
    # print(type(x))
    # print(y_)
    # print(np.array(y_))
    x = np.array(x) # 将list转为array格式，送进tf
    y_ = np.array(y_)   # 这里的y_里面放的是一个str
    y_ = y_.astype(np.int64)    # 转为int才能做label
    return x, y_

# 开始正式导入数据集
# 如果这些数据已经保存过了，直接读取即可
if os.path.exists(mnist_y_test_npy) and os.path.exists(mnist_x_test_npy) and os.path.exists(mnist_x_train_npy) \
    and os.path.exists(mnist_y_train_npy):
    print('-------------Load Datasets-----------------')
    # 读取npy文件，并且返回数组
    x_train_data = np.load(mnist_x_train_npy)
    y_train_data = np.load(mnist_y_train_npy)
    x_test_data = np.load(mnist_x_test_npy)
    y_test_data = np.load(mnist_y_test_npy)
    # 因为保存xdata时，把28*28的二维数组，转成一维保存了，所以这里读取的是60000*784的数据，需要强制转换为60000*28*28
    # print(x_train_data.shape)
    # print(y_test_data.shape)
    # ydata本身就是一列/一行，直接存不用变化了
    x_train_data = np.reshape(x_train_data, (len(x_train_data), 28, 28))
    x_test_data = np.reshape(x_test_data, (len(x_test_data), 28, 28))
    # print(x_train_data.shape)
# 如果这些数据没有初始化为npy格式，则自己读取一次，直接返回内存调用并存储到磁盘上，以便下次调用
else:
    print('-------------Generate Datasets-----------------')
    x_train_data, y_train_data = generate_dbs(mnist_train_jpg_path, mnist_tarin_label_path)
    x_test_data, y_test_data = generate_dbs(jpg_path=mnist_test_jpg_path, label_path=mnist_test_label_path)
    # print(x_train_data.shape, len(x_train_data))    # len (60000, 28, 28)只能取到最外层60000的长度
    # print(y_train_data.shape)

    print('-------------Save Datasets-----------------')
    x_train_save = np.reshape(x_train_data, (len(x_train_data), -1))
    x_test_save = np.reshape(x_test_data, (len(x_test_data), -1))
    np.save(mnist_x_train_npy, x_train_save)
    np.save(mnist_y_train_npy, y_train_data)
    np.save(mnist_x_test_npy, x_test_save)
    np.save(mnist_y_test_npy, y_test_data)
    # print(x_train_save.shape)

# 3.使用sequential构建model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# 4.model.compile
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 5.model.fit   validation_data必须用tuple传入，不能是list
model.fit(x=x_train_data, y=y_train_data, batch_size=32, epochs=5, validation_data=(x_test_data, y_test_data), validation_freq=1)

# 6.model.summary
model.summary()