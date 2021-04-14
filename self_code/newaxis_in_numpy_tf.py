import numpy as np
import tensorflow as tf

feature = np.array([[1, 2, 3],
                        [2, 4, 6]])
center = np.array([[1, 1, 1],
                   [0, 0, 0]])

print("原始数组大小：")
print(feature.shape)
print(center.shape)

np_feature_1 = feature[:, :, np.newaxis]  # 在末尾增加一个维度
np_feature_2 = feature[:, np.newaxis] # 在中间增加一个维度
np_center = center[np.newaxis, :] # 在首部增加一个维度

print("添加 np.newaxis 后数组大小：")
print(np_feature_1.shape)
print(np_feature_1)
print('-----')
print(np_feature_2.shape)
print(np_feature_2)
print('-----')
print(np_center.shape)
print(np_center)

tf_feature_1 = feature[:, :, tf.newaxis]  # 在末尾增加一个维度
tf_feature_2 = feature[:, tf.newaxis] # 在中间增加一个维度
tf_center = center[tf.newaxis, :] # 在首部增加一个维度

print("添加 tf.newaxis 后数组大小：")
print(tf_feature_1.shape)

print('-----')
print(tf_feature_2.shape)

print('-----')
print(tf_center.shape)