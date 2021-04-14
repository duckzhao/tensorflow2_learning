'''
传统循环网络RNN可以通过记忆体实现短期记忆，进行连续数据的预测，但是当连续数据的序列变长时，会使展开时间步过长，在反向传播更新参数时，
梯度要按照时间步连续相乘，会导致梯度消失。所以在1997年Hochreitere等人提出了长短记忆网络LSTM，通过门控单元改善了RNN长期依赖问题。
tf.keras.layers.LSTM()  实际在搭建的时候，和使用simpleRnn一样
units: Any, 记忆体的个数，类似于filters的个数概念
activation: str = 'tanh',
return_ sequences=False 仅最后时间步输出ht (默认)   一般最后一层用False，中间层用True,中间层输出提取的ht特征给下一层循环核，
然后下一层接着中间层的特征ht继续提取新的，更重要的特征，但最后一层没有下一层循环核了，所以就不需要输出中间ht记忆体，仅输出最后一个时间步的结果记忆体（答案）就行了。
LSTM层在fit的时候也对输入数据的维度有所要求，和simplernn一样训练集和测试集都必须np.reshape->【样本数，时间步展开步数，每个时间步的样本特征数】

用连续60天的股票价格，预测第61天的股票价格，输入序列是【60 天股票价格】---时间步展开数就是60，每个时间步的特征数是1
因为是做连续序列的数字预测，所以不用使用embedding进行编码，fit数据直接进LSTM层，所以train数据需要满足特征---【样本数，时间步展开步数，每个时间步的样本特征数】
'''
# 1.
import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # sklearn 数据预处理，归一化
from sklearn.metrics import mean_squared_error, mean_absolute_error     # skleran中的评价指标，计算均方误差和平均绝对误差
import math
np.set_printoptions(threshold=np.inf)

# 2.
maotai = pd.read_csv(filepath_or_buffer='./SH600519.csv')
# print(maotai.shape)
# print(maotai)
train_data = maotai.iloc[:2426-300, 2].values   # 返回无index和columns
test_data = maotai.iloc[2426-300:, 2].values
# print(train_data.shape)
# print(test_data.shape)
# 归一化
sc = MinMaxScaler(feature_range=(0, 1))  # 归一化到0-1之间
#根据对之前部分trainData进行fit的整体指标，对剩余的数据（testData）使用同样的均值、方差、最大最小值等指标进行转换transform(testData)，从而保证train、test处理方式相同。
train_data_sc = sc.fit_transform(train_data.reshape(-1, 1))    # 既包含了fit操作---求得输入数据的均值，最大值，最小值等属性，又包含了transformer利用这些属性都某个输入数据直接进行处理操作（归一化）
test_data_sc = sc.transform(test_data.reshape(-1, 1))  # 直接用train data绑定的train归一化指标对test数据进行相同处理，保证处理方式一致性

# 构造连续60天的序列样本数据,第61天的开盘价作为预测结果
# 构造训练集
x_train = []
y_train = []
for index in range(60, len(train_data_sc)):
    x_train.append(train_data_sc[index-60:index, 0])
    y_train.append(train_data_sc[index, 0])
# print(len(x_train))
# print(len(y_train))   # 2126-60=2066
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)
# 将训练集格式由list转为ndarray
x_train, y_train = np.array(x_train), np.array(y_train)
# 将训练集修改为符合 LSTM 层输入要求的维度
x_train = x_train.reshape((x_train.shape[0], 60, 1))
# print(x_train.shape)
# 构造测试集
x_test = []
y_test = []
for index in range(60, len(test_data_sc)):
    x_test.append(test_data_sc[index-60:index, 0])
    y_test.append(test_data_sc[index, 0])
# print(len(x_test))
# print(len(y_test))  # 300-60=240
x_test, y_test = np.array(x_test), np.array(y_test)
# 将测试集修改为符合 LSTM 层输入要求的维度
x_test = x_test.reshape((x_test.shape[0], 60, 1))

# 3.
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=80, return_sequences=True),  # 80个记忆体，一般最后一层才是false，前面的每层都要输出每个时间步的预测结果
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units=100, return_sequences=False),  # 最后一层rnn了
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)    # 因为是预测值，所以全连接只有一层
])

# 4.因为是预测数值，不是做分类，所以删去metrics选项
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')   # 使用均方误差损失函数

# 加载历史模型
check_point_save_path = './check_point/lstm_stock.ckpt'
if os.path.exists(check_point_save_path+'.index'):
    print('--------------load the model----------------')
    model.load_weights(filepath=check_point_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_point_save_path, save_best_only=True,
                                                 save_weights_only=True, monitor='val_loss')    # monitor为需要监视的值

# 5.
history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])

# 6.
model.summary()

# 打印参数名称，大小和内容
trainable_variables = model.trainable_variables
# print(trainable_variables)
with open(r'./rnn_weight.txt', mode='w')as f:
    for trainable_variable in trainable_variables:
        print(trainable_variable.name)
        print(trainable_variable.shape)
        f.write(str(trainable_variable.name)+'\n')
        f.write(str(trainable_variable.shape)+'\n')
        f.write(str(trainable_variable.numpy())+'\n')

# 画出loss和val loss曲线对比图
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title(label='Training and Validation Loss')
plt.legend()
plt.show()


################## predict ######################
# 将测试集手动预测一遍，并且画出预测值和实际值对比曲线
predict_test_stock_price = model.predict(x_test)
# 因为之前train data和test data都归一化过了，所以我们预测的结果验证值y也是归一化过的，需要还原回去
# 对预测值从0-1归一化范围返回至原始范围
predict_test_stock_price = sc.inverse_transform(predict_test_stock_price)
# 对真实值也返回原来范围
real_test_stock_price = sc.inverse_transform(y_test.reshape(-1, 1))
# 画出真实值和预测值的对比曲线
plt.plot(predict_test_stock_price, label='Predict Maotai Price', color='blue')
plt.plot(real_test_stock_price, label='Real Maotai Price', color='red')
plt.title(label='Real and Predict Stock Price')
plt.xlabel('Time')
plt.ylabel('Maotai Stock Price')
plt.legend()
plt.show()

##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(y_true=real_test_stock_price, y_pred=predict_test_stock_price)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mse)
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(real_test_stock_price, predict_test_stock_price)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)