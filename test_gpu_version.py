import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息

tensorflow_version = tf.__version__
gpu_available = tf.test.is_gpu_available()

print("tensorflow version:", tensorflow_version, "\tGPU available:", gpu_available)

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([1.0, 2.0], name="b")
result = tf.add(a, b, name="add")
print(result)
