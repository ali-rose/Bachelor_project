import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(0)

# 数据生成
n = 100  # 每类样本的数量
dimension = 100  # 数据维度
center1 = np.ones(dimension)  # 第一类中心
center2 = -np.ones(dimension)  # 第二类中心
X = np.zeros((2 * n, dimension))
Y = np.zeros(2 * n)
X[:n, :] = center1 + np.random.randn(n, dimension)
X[n:, :] = center2 + np.random.randn(n, dimension)
Y[:n] = 1
Y[n:] = -1

# 感知机模型初始化
w = np.zeros(dimension)
b = 0
learning_rate = 0.1
epochs = 10

# 感知机学习算法
for epoch in range(epochs):
    for i in range(2 * n):
        # 判断是否正确分类
        if Y[i] * (np.dot(w, X[i]) + b) <= 0:
            # 更新权重和偏置
            w += learning_rate * Y[i] * X[i]
            b += learning_rate * Y[i]

# 测试数据生成
m = 10  # 测试样本数
Xt = np.zeros((2 * m, dimension))
Yt = np.zeros(2 * m)
Xt[:m, :] = center1 + np.random.randn(m, dimension)
Xt[m:, :] = center2 + np.random.randn(m, dimension)
Yt[:m] = 1
Yt[m:] = -1

# 使用感知机模型进行测试
predictions = np.sign(np.dot(Xt, w) + b)
errors = predictions != Yt
error_rate = np.mean(errors)

# 输出错误率
print("Error rate:", error_rate)

# 绘制测试数据及分类界面
plt.figure(figsize=(7, 6))
plt.scatter(X[:n, 0], X[:n, 1], c='red', marker='o', edgecolors='k', label='class 1: train')
plt.scatter(X[n:, 0], X[n:, 1], c='blue', marker='*', edgecolors='k', label='class 2: train')
plt.scatter(Xt[:m, 0], Xt[:m, 1], c='green', marker='o', edgecolors='k', label='class 1: test')
plt.scatter(Xt[m:, 0], Xt[m:, 1], c='green', marker='*', edgecolors='k', label='class 2: test')

# 绘制分类界面
x_vals = np.linspace(min(X[:,0].min(), Xt[:,0].min()), max(X[:,0].max(), Xt[:,0].max()), 300)
y_vals = -(b + w[0] * x_vals) / w[1]
plt.plot(x_vals, y_vals, 'k--', label='classification boundary')

plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend()
plt.show()
