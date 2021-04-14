'''
可是使用plt.subplot在一副图上绘制多个子图，增加图形的对比效果
拆分子图及对指定图操作的规则如下：
plt.subplot 一直实际上都是对同一幅figure的不同子区域进行操作的，每次指定图的代码之间并无直接联系，如果不切换subplot，则一直对同一个subplot操作
plt.subplot(2, 2, 1)，就是这次选中子图的规则，将大图分为两行两列，然后当前操作的是序号为1的子图
plt.subplot(2, 2, 2) ，这次操作将大图依然分为两行两列，对序号为2的子图操作
plt.subplot(212)，这次操作把大图分为两行一列，（第一行的已经被前面两个子图填充了），对序号为2的 下面子图进行操作
plt.show() 展示大图

# 还可以使用如下参数，在subplot之前配置大图和子图排布规则
fig.tight_layout() # 调整整体空白
plt.subplots_adjust(wspace =0, hspace =0)  # 调整子图间距
# 参数
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots,
               # expressed as a fraction of the average axis width
hspace = 0.2   # the amount of height reserved for white space between subplots,
               # expressed as a fraction of the average axis height
'''

import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [5, 4, 3, 2]

plt.subplot(2, 2, 1)  # 呈现2行2列，第一行的第一幅图
plt.plot(x, y)

plt.subplot(2, 2, 2)  # 呈现2行2列，第一行的第二幅图
plt.barh(x, y)

plt.subplot(212)  # 第2行的全部
plt.bar(x, y)

plt.show()
