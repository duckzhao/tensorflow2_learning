'''
tf.TensorArray ：TensorFlow 动态数组，有点类似于python的list结构。不过在size上、存元素上 比list更加严格。

在部分网络结构，尤其是涉及到时间序列的结构中，我们可能需要将一系列张量以数组的方式依次存放起来，以供进一步处理。当然，在即时执行模式下，
你可以直接使用一个 Python 列表（List）存放数组，但这样就没法使用tf.function修饰该函数实现计算图特性加速，---静态图不支持翻译 py 的 list
因此，TensorFlow 提供了 tf.TensorArray ，一种支持计算图特性的 TensorFlow 动态数组。
'''

'''
其声明的方式为：
arr = tf.TensorArray(dtype, size, dynamic_size=False) ：声明一个大小为 size ，类型为 dtype 的 TensorArray arr 。
如果将 dynamic_size 参数设置为 True ，则该数组会自动增长空间。
读取和写入的方法分别为（除此以外，TensorArray 还包括 stack() 、 unstack() 等常用操作）:
write(index, value) ：将 value 写入数组的第 index 个位置；  并非原地返回，需要接收返回值
read(index) ：读取数组的第 index 个值；
'''

import tensorflow as tf

@tf.function
def array_write_and_read():
    # 实例化一个tf array对象，不开启自动增长空间
    arr = tf.TensorArray(dtype=tf.int32, size=3, dynamic_size=False)
    arr = arr.write(index=0, value=tf.constant(0))     # 可以传py 数字，也可以传tensor，如果要用计算图加速，最好传tensor
    arr = arr.write(index=1, value=tf.constant(1))
    arr = arr.write(index=2, value=tf.constant(2))    # 规定了arr类型是int，不能传float或者string类型的值
    arr = arr.write(index=0, value=3)   # 可以覆盖已有index二次写入
    # arr = arr.write(3, 4)     # 声明arr size的大小是3，且没开启自动增长，就不能超过size给arr写入元素
    a0 = arr.read(0)
    a1 = arr.read(1)
    a2 = arr.read(2)
    return a0, a1, a2

print(array_write_and_read())

@tf.function
def array_write_and_read1():
    # 实例化一个tf array对象，不开启自动增长空间
    arr = tf.TensorArray(dtype=tf.int32, size=3, dynamic_size=True)
    arr = arr.write(index=0, value=tf.constant(0))     # 可以传py 数字，也可以传tensor，如果要用计算图加速，最好传tensor
    arr = arr.write(index=1, value=tf.constant(1))
    arr = arr.write(index=2, value=tf.constant(2))    # 规定了arr类型是int，不能传float或者string类型的值
    arr = arr.write(index=0, value=3)   # 可以覆盖已有index二次写入
    arr = arr.write(3, 4)     # 声明arr size的大小是3，开启自动增长，可以超过size给arr写入元素
    arr = arr.write(5, 6)       # 且可以越值写入，不用沿着index递增写入，中间跳过的元素默认为0
    a0 = arr.read(0)
    a1 = arr.read(1)
    a2 = arr.read(2)
    a3 = arr.read(3)
    a4 = arr.read(4)    # 中间跳过的元素默认为0
    a5 = arr.read(5)
    return a0, a1, a2, a3, a4, a5

print(array_write_and_read1())