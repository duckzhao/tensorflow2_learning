'''
tf record是tf专用的一种数据存储格式，简单理解就是tf如果用tf record这种格式去读取文件，训练时，效率会非常的高，而且不会存在内存撑爆的现象。
比如我们现在训练数据时，都是用的fit去做的，但是这样有个问题就是fit时候需要传入x=train_data，这个traindata必须是完全加载在内存里面的，
虽然 我们一般可以通过 dataset先读全部图片路径，配合map，再当前batch实际用的时候再把当前batch的图片放到内存里面，来优化这一点；
或者可以自己写一个循环，把train data分为多份，每一份加载到内存后去单独fit，然后fit多次---（不好，单词训练样本减少，模型泛化力差）。
但是如果训练集以后特别大了，连全部图片地址内存都装不下了，就没法直接送到fit里面训练了。

另外传统分散数据集还有一个缺点就是，存了多个单独的小文件，这样tf读取的时候就很费io，很繁琐，但是tfrecord会把这些文件都处理成一个 单独磁盘二进制文件，
这样读取的时候只用一个io操作就可以了，增强了cpu效率，尤其对大型训练数据很友好。而且当我们的训练数据量比较大的时候，可以将数据分成多个TFRecord文件，来提高处理效率。

若数据读取和运算是不同步的【串行的】，那么意味着在完成了运算之后，需要进行IO来对硬盘上的数据进行读取，并将数据放入内存中，此时接着完成后续的运算，
由于这个过程中存在IO操作，造成大部分资源处于等待中，造成大量浪费，训练时间比较长。->普通的，图片地址dataset模式可以通过开启prefetch解决这个问题、
prefetch实际上可以理解为，多开了一个cpu线程，不停的给内存里 读取数据，然后每当gpu需要数据，直接从内存里拿就行了，加快了速度

总之在数据集较大时（计算机内存装不进去全部数据），先把数据集转为tfrecord格式是一定没错的，他对于训练的好处很多，解决了文件读取io和训练速率的问题。
* 似乎使用tfrecord读取数据是不占用内存的？不太明白，可能转为tfrecord之后，只用关心显存的大小去调节batchsize就行了。
* 对dataset的打乱最好在制作tfrecord数据集时同时进行，否则就得对tfrecord转dataset的对象进行shuffly，这样的话顺序性太强，数据集万一又大，
* shuffly的buffersize还是会撑爆内存空间。

tf record可以直接转为dataset格式（dataset = tf.data.TFRecordDataset(tfrecord_file)），结合map函数等预处理操作后，就能送进fit了。
'''


'''
tf record的格式如下：# dataset.tfrecords
TFRecord 可以理解为一系列序列化的 tf.train.Example元素（字典，一个example就是一条训练样本）所组成的列表文件，

[
    {   # example 1 (tf.train.Example)
        'feature_1': tf.train.Feature,  # 每个example里面的feature就是训练样本的值，包括 样本特征、标签等等
        ...
        'feature_k': tf.train.Feature   # 一个feature代表 样本的 一个属性
    },  （一个字典就是一个example，就是一条样本）
    ...
    {   # example N (tf.train.Example)
        'feature_1': tf.train.Feature,
        ...
        'feature_k': tf.train.Feature
    }
]
'''


'''
要注意将普通数据集处理为tfrecord格式本身也是一件耗费时间的工作，这里耗费的时间不一定抵得过使用tfrecord格式数据训练时减少的时间。处理过程如下：

1.先把数据集中的每个样本读取到内存；
2.将该元素转换为 tf.train.Example 对象（每一个 tf.train.Example 由若干个 tf.train.Feature 的属性组成，因此需要先建立Feature的字典）；
3.将该 tf.train.Example 对象序列化为字符串，并通过一个预先定义的 tf.io.TFRecordWriter 写入 TFRecord 文件。
所以当我们读取tfrecord数据时，读到的每个样本example都是序列化过的字符串了，不能直接用，需要我们用map函数完成反序列化才能用。
'''

'''
feature的构建规则如下，value值必须以列表包裹才能传：
tf.train.BytesList ：字符串或原始 Byte 文件（如图片），通过 bytes_list 参数传入一个由字符串数组初始化的 tf.train.BytesList 对象；
tf.train.FloatList ：浮点数，通过 float_list 参数传入一个由浮点数数组初始化的 tf.train.FloatList 对象；
tf.train.Int64List ：整数，通过 int64_list 参数传入一个由整数数组初始化的 tf.train.Int64List 对象。
'''

# 演示如何利用原始数据生成tfrecord数据，此时最好不要对原始数据进行预处理，保持原始数据的全部信息，转完tfrecord用的时候再用map预处理
# 案例中的代码依旧是先把所有图片路径读到内存再转为tfrecord的，小规模数据集可行，如果数据集特别大，可以使用多层循环，缓解内存中数据集地址压力
# 案例中读取时①没有打乱原始数据集顺序->在从原始数据集->tfrecord时，读图片路径时做；

import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt

# 1.读取训练集地址和标签到内存
data_dir = './fastai-datasets-cats-vs-dogs-2/train'
train_cats_filenames = [data_dir+'/cats/{}'.format(filename) for filename in os.listdir(data_dir+'/cats')]
train_dogs_filenames = [data_dir+'/dogs/{}'.format(filename) for filename in os.listdir(data_dir+'/dogs')]
# print(train_dogs_filenames)
train_filenames = train_cats_filenames + train_dogs_filenames
train_labels = [0] * len(train_cats_filenames) + [1] * len(train_dogs_filenames)    # 将 cat 类的标签设为0，dog 类的标签设为1
# print(len(train_filenames), len(train_labels))

# 2.乱序遍历图片地址，将图片特征读取到内存，再转为tfrecord格式，只运行一次就ok，所以注释掉
random_index = np.random.permutation(len(train_filenames))
tfrecord_file = data_dir+'/train.tfrecords'
# with tf.io.TFRecordWriter(tfrecord_file) as writer:     # 类似于with open，开启一个写文件的io
#     for index in random_index:
#         filename = train_filenames[index]
#         label = train_labels[index]
#         with open(filename, mode='rb') as f:
#             image = f.read()    # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
#         # 构建每个样本的feature字典
#         feature = {
#             'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象，str 二进制形式
#             'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))  # 标签是一个 Int 对象
#         }
#         # 将每个样本的feature合集字典，转为tfexample格式,注意这里的api都是Features
#         example = tf.train.Example(features=tf.train.Features(feature=feature))
#         # 将Example序列化为string，并写入 TFRecord 文件
#         writer.write(example.SerializeToString())


'''
我们可以通过以下代码，读取之前建立的 train.tfrecords 文件，并通过 Dataset.map 方法，使用 tf.io.parse_single_example 函数对数据集中的
每一个序列化的 tf.train.Example 对象解码完成反序列化，同时还可以在map方法中对每个样本做一些预处理，如统一图片大小之类的。

反序列化的时候需要有一个feature_description字典结构，用于告诉parse_single_example函数，dataset中每个样本的feature属性是什么样子的，这样
parse_single_example函数才能知道怎么解析当前传入的example_string。该字典的构成元素是 tf.io.FixedLenFeature

tf.io.FixedLenFeature 的三个输入参数 shape 、 dtype 和 default_value （可省略）为每个 Feature 的形状、类型和默认值。
这里我们的数据项都是单个的数值或者字符串，所以 shape 为空数组。
'''
# 将tfrecord文件直接加载为dataset对象
raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
# 定义Feature结构，告诉解码器每个Feature的类型是什么
feature_description = {
    'image': tf.io.FixedLenFeature(shape=[], dtype=tf.string),    # 这里我们的数据项都是单个的数值或者字符串，所以 shape 为空数组。
    'label': tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
}
# map操作 利用该函数 对从tfrecord中直接读出来的raw dataset对象，进行进一步解析，生成可以送给fit训练的dataset对象
# 在这里完成图片统一resize大小操作，此mao函数将raw dataset的单一样本属性example_string，转为两个一一配对的属性输出了
def _parse_example(example_string):
    # 这一句将序列化后的example_string转为序列化以前的 样本特征字典，如87行的样子,feature_dict有 image和label两个属性
    feature_dict = tf.io.parse_single_example(serialized=example_string, features=feature_description)
    # 此时的image属性是一个str byte类型的图片io对象，需要解码为训练用的tf tensor
    feature_dict['image'] = tf.io.decode_jpeg(contents=feature_dict['image'])
    # 这里还可以进行归一化操作其实，注意resize后如果不强转为uint8，可视化会乱码
    feature_dict['image'] = tf.image.resize(images=feature_dict['image'], size=[256, 256])
    return feature_dict['image'], feature_dict['label']

# 此时的dataset中 每个样本有两个属性可以遍历，这个dataset是可以直接送进fit的
dataset = raw_dataset.map(_parse_example)

# 展示下resize后的图片
for image, label in dataset:
    plt.title('dog' if label else 'cat')
    # 强转为正常像素，否则resize后无法可视化
    plt.imshow(tf.cast(image, tf.uint8).numpy())
    plt.show()
    break
