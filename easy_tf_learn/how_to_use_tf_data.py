'''
tf.data 帮助我们建立属于自己的一套数据集，构建从原始数据输入->tf中需要的格式的数据，并且附带batch的生成方式
tf.data 的核心是 tf.data.Dataset 类，提供了对数据集的高层封装。
'''

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

'''
from_tensor_slices 适用于数据量较小（能够整个装进内存）的情况，输入样本和标签 在0维方向进行叠加，类似于[样本1特征，样本2特征，---，样本n特征]
# data = tf.data.Dataset.from_tensor_slices((输入特征，标签))  --->传入一个元组
# (Numpy和Tensor格式都可用该语句读入数据)
features 和 labels的第0维大小必须相同，才能一一对应上
'''
features = tf.constant(value=np.arange(12), shape=[3, 4])   # 样本特征
labels = tf.constant(value=[1, 2, 3])   # 样本标签
dataset = tf.data.Dataset.from_tensor_slices((features, labels))    # 聚合为一个组合带标签样本---tensor格式的
print('构建的数据集为：', dataset)
'''
<TensorSliceDataset shapes: ((4,), ()), types: (tf.int32, tf.int32)>
表明每一个样本是一维长度为4的向量， 数值标签
'''
for element in dataset:
    print('样本的内容为：', element)

# 使用tf.data.Datasets载入minst数据集
(train_x, train_y), (_, _) = tf.keras.datasets.mnist.load_data()
train_x = tf.cast(train_x, dtype=tf.float32)/255.    # 转换数据类型float32
train_x = train_x[:, :, :, tf.newaxis]      # 在末尾新增一个维度, [60000, 28, 28, 1]
# print(train_x[0])
mnist_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))  # 第0维都是60000长度
# 将原始数据集转换为特征和标签1 1 对应的数据
for image, label in mnist_dataset:
    plt.title(label.numpy())
    plt.imshow(image.numpy())
    plt.show()
    break

'''
对于特别巨大而无法完整载入内存的数据集，我们可以先将数据集处理为 TFRecord 格式，然后使用 tf.data.TFRocordDataset() 进行载入。
'''

'''
tf.data.Dataset 类为我们提供了多种数据集预处理方法。
'''
# 1.Dataset.map(f) 对数据集中的每个元素应用函数 f ，得到一个新的数据集（这部分往往结合 tf.io 进行读写和解码文件， tf.image 进行图像处理）；
# f函数的参数应该和当前的dataset迭代对象保持一致，比如用mnist数据集生成dataset对象时时，每一个dataset迭代对象是一个tensor，他的构成是 特征
# 和标签，所以dataset.map（f）这个f就需要有两个参数，一个参数接收特征，一个参数接收label。并且最好返回两个参数，保持源数据集格式不变化，但可以
# 对传入的特征在f中进行处理，比如图片增强，旋转，二值化等等。        这样处理的好处是批操作。
def rot90(image, label):
    image = tf.image.rot90(image)
    return image, label
# 返回一个新的数据集，格式和原来数据集保持一致，因为我们f函数 输入参数和输出参数是一致的
mnist_dataset1 = mnist_dataset.map(map_func=rot90)
for image, label in mnist_dataset1:
    plt.title(label.numpy())
    plt.imshow(image.numpy())
    plt.show()
    break

# 使用 Dataset.batch() 将数据集划分批次，每个批次的大小为 4：划分后每个迭代对象中的样本数为batch个（变成一个张量了）
# 这样做的目的在于我们构建自己的model时，如果想采用自写epoch和batch_size循环去训练model，每次给model喂入batch_size时都要自己写一个load_data
# 函数，来加载指定batch_size的数据，比较麻烦，我们可以直接使用dataset1 = dataset.batch(batch_size)来自动分组数据集，然后只需要遍历dataset1
# 中的子元素，batch_size大小的张量特征和标签元组对，直接喂入model进行训练即可，比较简单
mnist_dataset1 = mnist_dataset.batch(4)
for images, labels in mnist_dataset1:     # image: [4, 28, 28, 1], labels: [4]
    for index in range(4):
        plt.subplot(1, 4, index+1)
        plt.title(labels.numpy()[index])
        plt.imshow(images.numpy()[index])
    plt.show()
    break

# tf.random.set_seed(111)   如果指定了全局的yf random seed，则每次shuffle的结果一致

# 使用 Dataset.shuffle() 可以将数据随机乱序，这样可以保证每批次训练的时候所用到的数据集是不一样的，可以提高模型训练效果
# shuffle的功能为打乱dataset中的元素，它有一个参数buffersize，表示打乱时使用的buffer的大小，不设置会报错，
# buffer_size=1:不打乱顺序，既保持原序， buffer_size越大，打乱程度越大
# 注意：shuffle的顺序很重要，应该先shuffle再batch，如果先batch后shuffle的话，那么此时就只是对batch进行shuffle，而batch里面的数据顺序依旧是有序的，那么随机程度会减弱。
# 如下所示，先以1w的混乱度打散原始数据集中的标签对，然后再按照每批4个进行分组，得到mnist_dataset1
mnist_dataset1 = mnist_dataset.shuffle(buffer_size=10000).batch(4)
for images, labels in mnist_dataset1:
    for index in range(images.shape[0]):
        plt.subplot(1, 4, index+1)
        plt.title(labels.numpy()[index])
        plt.imshow(images.numpy()[index])
    plt.show()
    break

# 使用 dataset.repeat(count=None) 使得dataset中的数据集重复count次，实际上就是直接复制count次，主要用于完成epoch重复功能
temp_dataset = tf.data.Dataset.from_tensor_slices((np.arange(12), np.arange(12, 24)))
temp_dataset = temp_dataset.repeat(count=2)
for i, j in temp_dataset:   # i,j都是标量tensor
    print(i, j)
    break

