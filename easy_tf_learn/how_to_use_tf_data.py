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

# shuffle中的buffer_size不应该设置的过大，否则会造成计算机内存ram溢出(Shuffle buffer filled)，shuffly主要起到打乱作用，如果原始数据顺序较强，且数据集很大，
# 则不考虑用shuffly打乱，而使用 random_index = np.random.permutation(shape),使用这个index进行打乱，cats_vs_dogs_classify_vgg_model.py
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


'''一般而言我们在批训练模型时，都是使用map、shuffle、batch后的mnist_dataset1遍历进行的，但实际上map这类操作在执行这一行代码的时候并没有就对
原始的数据集mnist_dataset直接处理，而是定义了这样一个处理方式，真正的处理是在我们使用 for循环遍历提取 mnist_dataset1 中的数据时处理的，
这个时候就有一个问题，准备数据是cpu的工作，训练数据的gpu的工作，这默认是一个串行过程，准备数据时gpu处于idle状态，降低了gpu的利用率，和训练速度
因此如果我们能把准备输出时的cpu和训练的gpu变为并行过程，就不会浪费了.
Dateset对象提供了一个prefetch()方法，使得我们可以让数据集对象 Dataset 在训练时预取出若干个元素，使得在 GPU 训练的同时 CPU 可以准备数据，
使用过程非常简单，仅需在Dataset对象 提取、遍历送入训练前加上一句，其中buffer_size当前表示由 TensorFlow 自动选择合适的数值：
mnist_dataset = mnist_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
'''
# 如对82行代码做如下改变
mnist_dataset1 = mnist_dataset.shuffle(buffer_size=10000).batch(4)
mnist_dataset1 = mnist_dataset1.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # 开启数据预加载
for images, labels in mnist_dataset1:       # 预加载的数据在这一行被送入
    for index in range(images.shape[0]):
        plt.subplot(1, 4, index+1)
        plt.title(labels.numpy()[index])
        plt.imshow(images.numpy()[index])
    plt.show()
    break


'''
对上面的Dataset数据准备过程再进行深入理解，实际上cpu仅执行io操作或者一些图像处理操作时有用，map虽然是预定义的，但实际在110行数据加载时候，
才对当前批的读入数据进行处理，因此map操作是可以利用gpu资源的，只需要加入这句代码，其中2参数可以修改为tf.data.experimental.AUTOTUNE
'''
mnist_dataset = mnist_dataset.map(map_func=rot90, num_parallel_calls=2)


'''
对于构建好的Dataset数据集的使用：dataset返回的是一个py的可迭代对象，所以可以通过for来使用，要注意的是from_tensor_slices方法传入的tuple可以
不止两个元素，多个也可以，该函数仅作横向拼接，类似于zip效果，因此可以使用如下调用方式
dataset = tf.data.Dataset.from_tensor_slices((A, B, C, ...))
for a, b, c, ... in dataset:
    # 对张量a, b, c等进行操作，例如送入模型进行训练
也可以使用 iter() 显式创建一个 Python 迭代器并使用 next() 获取下一个元素，不过不如for循环直观
dataset = tf.data.Dataset.from_tensor_slices((A, B, C, ...))
it = iter(dataset)
a_0, b_0, c_0, ... = next(it)
a_1, b_1, c_1, ... = next(it)
'''

'''
实际上在使用model.fit时，我们可以不用x=train_x, y=train_y这样传入，可以直接传入一个包含x和y的dataset (x,y) 格式作为x，忽略y的传入，
同时因为我们已经在前面指定了batch，所以fit参数里也不用写batch_size参数了，这样的好处是①提高cpu、gpu的利用率；②不用一次加载全部数据（可以使用
tf的io操作，结合map从路径数据集中按批加载训练数据，不会内存溢出）------当然如果dataset已经把图片全部加载到内存里了，map部分的分批处理也能节约内存.

传入方式化简如下：原始mnist  model.fit(x=train_data, y=train_label, epochs=num_epochs, batch_size=batch_size)
化简后mnist： model.fit(mnist_dataset, epochs=num_epochs)

p.s. 再使用evaluate()函数传入测试集评估model时，也可以直接传入dataset对象
'''


'''
p.s. 接下来 基于 tf.data 和 Dataset 对象完成猫狗二分类的任务 ，见 cats_vs_dogs_classify_xxx_model.py
'''