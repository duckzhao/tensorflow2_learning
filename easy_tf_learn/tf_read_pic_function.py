'''
展示使用tensorflow读取图片的多种方式
'''
import tensorflow as tf
from matplotlib import pyplot as plt

def read_singel_pic(filepath):
    # 返回str形式的 二进制编码 的文件内容数据
    image_str = tf.io.read_file(filename=filepath)
    # print(image_str)  b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01‘
    # 对于图片数据，可以在tf.image下找到相应的解码函数，将io读入的string二进制数据转为 图片像素的uint8 tensor
    image_decoded = tf.image.decode_jpeg(contents=image_str)

    # 还可以对 image_decoded 进行resize或者 归一化/255. 等操作后再return，常用于map函数与dataset配合中
    return image_decoded

if __name__ == '__main__':
    image_decoded = read_singel_pic('./10.jpg')
    plt.imshow(image_decoded.numpy())
    plt.show()