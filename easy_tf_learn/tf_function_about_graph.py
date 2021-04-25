'''
tf2所采用的eager execution，即动态图、即时模式虽然使得我们能够更快的调试代码，但是在实际的生产环境中显然动态图不如静态图运算快，
因此为了追求效率，tf也提供了一个简单的机制用于将动态图转换为TensorFlow 1.X 中默认的图执行模式（Graph Execution）
TensorFlow 2 为我们提供了 tf.function 模块，结合 AutoGraph 机制，使得我们仅需加入一个简单的 @tf.function 修饰符，就能轻松将模型以图执行模式运行
'''

'''
tf.function基础用法：在 TensorFlow 2 中，推荐使用 tf.function （而非 1.X 中的 tf.Session ）实现图执行模式，
从而将模型转换为易于部署且高性能的 TensorFlow 图模型。
只需要将我们希望以图执行模式运行的代码封装在一个函数内，并在函数前加上 @tf.function 即可。(好像不能修饰类函数，只能修饰普通函数？)
在被修饰的代码首次执行时，tf会在缓存的哈希表中检查有没有该函数的缓存，如果没有，就按行执行该函数，并且把该函数逐句、静态编译为tf的原生代码。
在后续调用该函数时，就会在缓存表里找这个函数，如果有则执行缓存表中翻译的tf函数（省略非tf原生代码），如果没有则翻译该函数到哈希表缓存。

因此被修饰的函数不能有太复杂的python语句，函数参数最好只有tf或者np类型的参数，if for可以有，被翻译为tf中的对应语句，但print不能有
'''
import tensorflow as tf
import numpy as np

@tf.function
def f(x):
    print("The function is running in Python")
    tf.print(x)

# tf格式的输入或者np格式的输入时，缓存图工作原理
a = tf.constant(1, dtype=tf.int32)  # 第一次，哈希表无缓存，全部执行并编译为tf语句，输出print
f(a)
b = tf.constant(2, dtype=tf.int32)  # 第二次，哈希表有缓存，仅执行翻译的原生tf语句，无print
f(b)
b_ = np.array(2, dtype=np.int32)    # 第三次，虽然参数变了，但是tf会把np自动转为tf格式，等于没变参数类型，因此同第二次执行结果
f(b_)
c = tf.constant(0.1, dtype=tf.float32)  # 第四次，参数类型变了，哈希表无缓存，重新全部执行，并翻译输入参数为float32时候的缓存，有print
f(c)
d = tf.constant(0.2, dtype=tf.float32)  # 第五次，参数类型float32，查表有缓存，因此仅执行tf原生语句，无print
f(d)


# python原生数据类型-整数、浮点数，不会被自动转为tf.constant，因此只有输入的数字完全一样的时候才会触发缓存，否则即使都是 整数（1，2）也会全部执行
'所以在给tf函数传参时，最好将参数转为tf的数据类型再传入---不管该函数是否被@tf.function修饰'
f(1)    # 第一次，全部输出，记录 py整数 参数1 的缓存
f(2)    # 第二次，全部输出，记录 py整数 参数2 的缓存
f(1)    # 第三次，哈希表查到 py整数 参数1 的缓存，不输出print
f(0.1)  # 第四次，全部输出，记录 py浮点数 参数0.1 的缓存
f(0.2)  # 第五次，全部输出，记录 py浮点数 参数0.2 的缓存
f(0.1)  # 第六次，哈希表查到 py浮点数 参数0.1 的缓存，不输出print