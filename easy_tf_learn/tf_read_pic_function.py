'''
展示使用tensorflow读取图片的多种方式
* 小心坑：tf.image.resize在对图像resize后，会造成像素越界的情况，其直接反应就是 resize后返回的 image_resized已经无法使用plt绘图了，
* 画出来的图片完全失真。resize后的tensor类型为float32，而像素要求是 uint8。如果我们还希望对resize后的图片进行可视化，则应在return前加上
* tf.cast完成强制类型转换，例read_singel_pic_0。

如果这一步是应用于处理训练集，且马上就要归一化/255.了，则不用太担心，
'''
import tensorflow as tf
from matplotlib import pyplot as plt

# 仅仅完成resize图片功能，且要返回可以可视化的像素数组tensor
def read_singel_pic_0(filepath):
    # 返回str形式的 二进制编码 的文件内容数据
    image_str = tf.io.read_file(filename=filepath)
    # print(image_str)  b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01‘
    # 对于图片数据，可以在tf.image下找到相应的解码函数，将io读入的string二进制数据转为 图片像素的uint8 tensor
    image_decoded = tf.image.decode_jpeg(contents=image_str)
    print('转换前：', image_decoded.dtype, image_decoded.shape)
    image_decoded = tf.image.resize(image_decoded, size=[256, 256])
    print('转换后：', image_decoded.dtype, image_decoded.shape)  # 变为float32类型，且存在像素溢出问题
    return tf.cast(image_decoded, dtype=tf.uint8)   # 强转为正常像素


# 用于训练model时对dataset中的 filepath数据读取，并归一化为训练像素，适用map函数，无法可视化
def read_singel_pic_1(filepath, label):
    # 返回str形式的 二进制编码 的文件内容数据
    image_str = tf.io.read_file(filename=filepath)
    # print(image_str)  b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01‘
    # 对于图片数据，可以在tf.image下找到相应的解码函数，将io读入的string二进制数据转为 图片像素的uint8 tensor
    image_decoded = tf.image.decode_jpeg(contents=image_str)
    # 完成resize，顺便归一化，常用于map函数与dataset配合中
    image_resized = tf.image.resize(image_decoded, size=[256, 256]) / 255.
    # 切记要加上这一行，否则class的model无法检测到图片channel
    image_resized.set_shape([256, 256, 3])
    return image_resized, label


if __name__ == '__main__':
    image_decoded = read_singel_pic_0('./10.jpg')
    plt.imshow(image_decoded.numpy())
    plt.show()
