'''
本代码为简单粗暴的tensorflow2中的训练猫狗案例代码，仅对原始代码做了打乱数据处理，其余基本保持不变，此代码由于没有测试集
model checkpoint采用monitor监视，另外shuffly的batchsize不能设置过大，这是一个坑，但是也必须要打乱原始数据集，否则训练过拟合，
可以采用np.random.permutation方式生成打乱traindata的索引，然后numpy格式取索引切片，完成训练集的重新乱序。
由于案例中的sequential model比较简单，因此测试集准确率只有63左右，训练集准确率有90+，有过拟合现象。
'''
import tensorflow as tf
import os
import numpy as np

num_epochs = 10
batch_size = 128
learning_rate = 0.001
data_dir = './fastai-datasets-cats-vs-dogs-2/'
train_cats_dir = data_dir + '/train/cats/'
train_dogs_dir = data_dir + '/train/dogs/'
test_cats_dir = data_dir + '/valid/cats/'
test_dogs_dir = data_dir + '/valid/dogs/'

def _decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)            # 读取原始文件
    image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片
    image_resized = tf.image.resize(image_decoded, [256, 256]) / 255.0
    return image_resized, label

if __name__ == '__main__':
    # 构建训练数据集
    train_cat_filenames = tf.constant([train_cats_dir + filename for filename in os.listdir(train_cats_dir)])
    train_dog_filenames = tf.constant([train_dogs_dir + filename for filename in os.listdir(train_dogs_dir)])
    train_filenames = tf.concat([train_cat_filenames, train_dog_filenames], axis=-1)
    train_labels = tf.concat([
        tf.zeros(train_cat_filenames.shape, dtype=tf.int32),
        tf.ones(train_dog_filenames.shape, dtype=tf.int32)],
        axis=-1)

    # 同时打散训练集路径和训练集标签
    # 先生成乱序的随机索引
    random_index = np.random.permutation(train_filenames.shape[0])
    # 通过numpy 切片与索引的方式 重新取乱序值
    train_filenames = train_filenames.numpy()[random_index]
    train_labels = train_labels.numpy()[random_index]

    # 将乱序后的ndarray数据构造Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(
        map_func=_decode_and_resize,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # 取出前buffer_size个数据放入buffer，并从其中随机采样，采样后的数据用后续数据替换---前5k个都是猫，只有补充的少数dog，一开始肯定过拟合。需要重新打散，更散
    # 但是如果把buffer_size设为23000，电脑内存又顶不住，因此，我们考虑在train_filenames处进行打散，字符串比较小，方便打散
    train_dataset = train_dataset.shuffle(buffer_size=2000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 5, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

    # 再fit之前设置保存model，并且判断能否加载历史训练参数
    check_point_path = './fastai-datasets-cats-vs-dogs-2/models/VGGNet_cat_dog.ckpt'
    if os.path.exists(check_point_path + '.index'):
        print('-------------load the mode1-------------')
        model.load_weights(check_point_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_point_path, save_weights_only=True,
                                                     save_best_only=True, monitor='loss')
    # 保存logs日志，用于tensorboard可视化
    tensorboard_log_path = './cat_dog_logs/'
    # 如果该地址不存在就创建一个logs文件夹
    if not os.path.exists(tensorboard_log_path):
        os.mkdir(tensorboard_log_path)
    # 可以每个batch保存一次loss或者别的信息，指定 updata_freq='batch'即可
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_path, histogram_freq=1, update_freq='batch')

    model.fit(train_dataset, epochs=num_epochs, callbacks=[cp_callback, tensorboard_callback])