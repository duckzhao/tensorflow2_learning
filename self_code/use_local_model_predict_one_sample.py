'''
当本地已经有训练好的model参数时，可以直接加载这个model参数，完成对新样本的预测任务---前向传播预测
1.复现模型，可以使用Sequential或者class---但必须和以前生成模型时一致
2.加载参数---model.load_weights(path)   .ckpt
3.执行预测---model.predict(x_sample)---x_sample的维数，要和训练时传入的保持一致（如果训练用了batch_size）则单个样本需要新增第一个维度
p.s. 预测返回值也同以前网络输出一样，（比想象中多一个维度，二维向量）多一个batch_size的概率输出，需要argmax转换下
'''

import tensorflow as tf
import numpy as np
from PIL import Image

# 1.
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


# 2.
model.load_weights(filepath='./check_point/mnist.ckpt')

# 3.
def predict_sample(input_path):
    # 输入预处理
    img = Image.open(input_path)
    # img.show()

    # 由于输入图像大小不一定是28*28因此强转大下为28*28   Image.ANTIALIAS指定生成图片是高质量的
    img = img.resize((28, 28), Image.ANTIALIAS)
    # img.show()

    # 将 PngImageFile 格式的img转为像素数组,此时的img是三通道图片，我们需要的是单通道图片，再做强转
    img_arr = np.array(img.convert('L'))
    # print(img_arr.shape)    # (28, 28)，不加convert的结果是(28, 28, 3)
    # img_ = Image.fromarray(img_arr)
    # img_.show()

    # 预处理：又由于训练时的图片是黑色背景白色字，而测试集的图片是白色背景灰色字，因此做一个像素转换 白色255， 黑色0
    img_arr = 255 - img_arr # 这样白色背景变黑，灰色字体变白---即颜色取反
    # 还可以进一步将数字图片转为仅有 纯黑纯白两色的黑白图片
    for i in range(28):
        for j in range(28):
            if img_arr[i][j] < 30:  # 如果该像素点偏黑
                img_arr[i][j] = 0   # 转为黑色点
            else:
                img_arr[i][j] = 255  # 转为白色点

    # img_ = Image.fromarray(img_arr)
    # img_.show()

    # 按照神经网络输入前的特征归一化
    img_arr = img_arr/255.
    # print(img_arr.shape)  # (28*28)
    # 我们神经网络输入需要的是(N*28*28),因此新增一个维度,表示在最前面插入一个维度
    img_arr = img_arr[tf.newaxis, :, :]
    # 返回一个二维数组，因为只有一个样本，因此shape是(1, 10),否则是（N,10）
    result = model.predict(x=img_arr)
    print(result)
    label = result.argmax(axis=1)
    # print(label)
    return label

if __name__ == '__main__':
    for i in range(10):
        label = predict_sample(input_path='./MNIST_FC/{}.png'.format(i))
        print(label)
    label = predict_sample(input_path='me_6.png')
    print(label)