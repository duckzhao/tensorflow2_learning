'''
使用vgg16完成迁移学习，对dog cat二分类任务进行小样本训练
迁移学习主要指  加载在大训练集上已经训练的不错的模型，由于 深度学习模型的前几层都是提取边角特征，最后几层才是结合任务内容完成图像处理
所以我们可以 对前面的预训练网络保持不变(主要是前面的卷积层保持不变，设置include_top=False后，后面的flatten和全连接层dense都没有了)，
然后我们手动添加 flatten 和 dense层即可，设置预训练model中的layers为不可训练模式，仅训练后面的几层全连接，实现小样本学习
'''

import tensorflow as tf
import os
import cv2
import numpy as np

# 处理数据
image_path = r'C:\Users\Z\Desktop\images'
dirs = ['\cats', '\dogs']
image_path_list = []
image_label_list = []
for dir in dirs:
    data_path = os.listdir(image_path+dir)
    image_path_list.extend([image_path+dir+'\{}'.format(__) for __ in data_path])
    # 猫 0.狗 1
    if dir == '\cats':
        image_label_list.extend([0]*len(data_path))
    else:
        image_label_list.extend([1] * len(data_path))
print(image_path_list)
print(image_label_list)


train_x = []
train_y = image_label_list
for __ in image_path_list:
    img = cv2.imread(__)
    # 统一大小为 224, 224, 3
    img = cv2.resize(img, dsize=(224, 224))
    img = img.astype(np.float32)
    # 归一化
    img = img / 255.
    train_x.append(img)

# 转为 array
train_x = np.array(train_x)
train_y = np.array(train_y)
# 打乱顺序
shuffle_ix = np.random.permutation(np.arange(len(train_y)))
train_x = train_x[shuffle_ix]
train_y = train_y[shuffle_ix]

weights_path = r'C:\Users\Z\Desktop\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
# 加载预训练的 VGG16 模型，但取消最后几层全连接
vgg_16_model = tf.keras.applications.VGG16(include_top=False, weights=weights_path, input_shape=(224, 224, 3))
# 冻结预训练网络的所有layer，设置为不可训练模式
for layer in vgg_16_model.layers:
    layer.trainable = False
# 打印 删减后 预训练网络的结构
vgg_16_model.summary()

# 给预训练网络 新增一些当前任务的层
# 将最后池化的(None, 7, 7, 512) 拉直
model = tf.keras.layers.Flatten()(vgg_16_model.output)
model = tf.keras.layers.Dense(units=1024, activation='relu')(model)
model = tf.keras.layers.Dense(1024, activation='relu')(model)
model = tf.keras.layers.Dropout(0.3)(model)
model = tf.keras.layers.Dense(2, activation='softmax')(model)

# 将预训练网络和自定义网络结合
final_model = tf.keras.Model(inputs=vgg_16_model.input, outputs=model)

# 打印组合后的网络结构
final_model.summary()

# 配置网络训练方式
final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['sparse_categorical_accuracy'])

# 存储历史模型、加载历史模型
model_save_path = './transfer_check_point/vgg16_transfer_model.ckpt'
if os.path.exists(model_save_path+'index'):
    print('--------------------load the model---------------------')
    final_model.load_weights(model_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(model_save_path, monitor='val_loss',
                                                 save_weights_only=True, save_best_only=True)

history = final_model.fit(train_x, train_y, batch_size=100, epochs=100, callbacks=[cp_callback], validation_split=0.2,
                          shuffle=True, validation_freq=1)