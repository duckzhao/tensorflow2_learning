'''
开始学习RNN的第一课代码，首先介绍几个概念---
1.循环核：循环核是循环神经网络的根本，类似于cnn中的卷积层一样的概念，
2.记忆体：每个循环核可以指定多个记忆体，类似于cnn中的filters概念，记忆就是ht---记忆体的个数是否为ht矩阵的个数？ht是隐藏层的输出
3.时间步：因为在RNN里面我们输入的是“序列数据”，所以每一层循环核的计算都是横向读取序列数据的，如序列 a b c d e作为一个样本输入的序列数据，在一个循环核的计算过程里就会有5个时间步。
4.循环核时间展开步数：序列 a b c d e作为一个样本输入的序列数据，每个时间步都输出一个结果的话，就是a->结果 ；b->结果···，循环核时间展开步数就是1
如果是 a b c d ->e，如果四个时间步才输出一个结果，循环核时间展开步数就是4。并不是所有时间步都有输出的，取决于循环核时间展开步数。---（是否没输出就不能反向传播更新，4步才能更新一次？）
5.循环计算层的层数：即循环核（层）的个数是可以按照输出方向增长的，即一个循环计算层可以纵向放多个循环层，类似于cnn的叠加卷积层一样
6.参数矩阵共享：rnn的循环体现在横向时间序列的计算上，参数矩阵在同一个序列sample横向展开是是不变的，然后ht（记忆）随着序列的输入，核参数矩阵做运算，
不断改变，然后送给下一层ht，和下一层输入的序列中的元素进行类似运算，然后再把ht送给下一层（横向的下一层，应该说送入下一个序列元素的输入时间步）。
在纵向上循环层的参数矩阵并不共享，进行较为独立的运算，提取特征，不体现序列性。
-------------------------------------------------------------------------------------------------
了解RNN的最基本api:tf.keras.layers.SimpleRNN()
units: 该层循环核中记忆体的个数（类似于filters个数），越多特征提取越准，但耗费资源
activation: str = 'tanh',
return_sequences= 是否每个时刻输出ht到下一层。一般最后一层的循环核用False，仅在最后一个时间步输出ht；中间的层循环核用True，每个时间步都把ht输出给下一层。（ht就是对序列数据的记忆）
return_ sequences=False 仅最后时间步输出ht (默认)   一般最后一层用False，中间层用True,中间层输出提取的ht特征给下一层循环核，
然后下一层接着中间层的特征ht继续提取新的，更重要的特征，但最后一层没有下一层循环核了，所以就不需要输出中间ht记忆体，仅输出最后一个时间步的结果记忆体（答案）就行了。

SimpleRNN在fit时对送入的训练集维度是有要求的---[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]。送入样本数（len(x_train)即可），
循环核时间展开步数见概念4，每个时间步输入特征个数实际上就是样本的特征数
x_train = np.reshape(x_train, (len(x_train), 1, 5))
y_train = np.array(y_train)  y_train和以前一样保持是数字向量即可

使用六步法实现最基本的RNN训练序列数据结构---字母预测，1per1
'''

# 1.
import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)

# 2.
input_word = 'abcde'
word_to_id = {input_word[index]: index for index in range(len(input_word))}
# print(word_to_id)
# 因为神经网络用到的都是数字，所以我们要把字母用数字向量表示出来，也就是字母映射到词向量数字空间，最简单的就是one-hot编码，一个词对应一个空间，但浪费大
id_to_onehot = {0: [1., 0., 0., 0., 0.], 1: [0., 1., 0., 0., 0.], 2: [0., 0., 1., 0., 0.], 3: [0., 0., 0., 1., 0.],
                4: [0., 0., 0., 0., 1.]}  # id编码为one-hot,这个就是输入向量，得转成float，不然tensor计算有问题
x_train = [id_to_onehot[word_to_id['a']], id_to_onehot[word_to_id['b']], id_to_onehot[word_to_id['c']],
           id_to_onehot[word_to_id['d']], id_to_onehot[word_to_id['e']]]
y_train = [word_to_id['b'], word_to_id['c'], word_to_id['d'], word_to_id['e'], word_to_id['a']]
# 打乱数据集
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)
# 转xtrain的维度符合simpleRNN fit要求，[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
# 循环核时间展开步数:输入一个字母，预测一个字母，所以是1---每个时间步输入特征个数：每个字母都是5个特征，所以是5
x_train = np.reshape(x_train, (len(x_train), 1, 5))
y_train = np.array(y_train)
# print(x_train, y_train)   # 顺便转为numpy格式

# 3.
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(units=5),  # 一个循环核（层），里面有三个记忆体，只有一层即最后一层，return_sequences=False
    tf.keras.layers.Dense(units=5, activation='softmax')
])

# 4.
model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
# 加载历史模型
check_point_save_path = './check_point/rnn_onehot_1per1.ckpt'
if os.path.exists(check_point_save_path+'.index'):
    print('--------------load the model----------------')
    model.load_weights(filepath=check_point_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_point_save_path, save_best_only=True,
                                                 save_weights_only=True, monitor='loss')    # monitor为需要监视的值

# 5.
# 由于fit没有给出测试集，不计算测试集准确率，[根据loss，保存最优模型],一般cp_callback里面都有默认的val_loss，即验证集损失保存模型，这里没有验证集，
# 所以就不需要写 validation_xxx的参数，不做验证，同时采用测试集loss保存最优模型
history = model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=[cp_callback])

# 6.
model.summary()

# 打印参数名称，大小和内容
trainable_variables = model.trainable_variables
print(trainable_variables)
with open(r'./rnn_weight.txt', mode='w')as f:
    for trainable_variable in trainable_variables:
        print(trainable_variable.name)
        print(trainable_variable.shape)
        f.write(str(trainable_variable.name)+'\n')
        f.write(str(trainable_variable.shape)+'\n')
        f.write(str(trainable_variable.numpy())+'\n')

# acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Loss')
plt.title('Loss')
plt.legend()

plt.show()

while True:
    word = input('please input a word, exit input esc!')
    if word == 'esc':
        break
    else:
        # 将word转为数字向量
        word_ = word
        word = id_to_onehot[word_to_id[word]]
        # 将向量转为符合fit适应的三维矩阵,[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
        word = np.reshape(word, (1, 1, 5))
        # 预测的时候都需要给传入数据新增一个维度的，因为训练的时候都是批训练
        # result = model.predict(x=word[np.newaxis, :, :])
        # print(word.shape)
        # print([word])
        # print(np.array([word]).shape)
        # print(word[np.newaxis, :, :].shape)
        # print(tf.constant(word[np.newaxis, :, :]).shape)
        # print(tf.constant([word]).shape)

        result = model.predict(x=tf.constant(word))
        # result -> [[1, 2, 3, 4, 5]]
        result = tf.argmax(result, axis=1)
        tf.print('input {} -> {}'.format(word_, result[0]))