import numpy as np
import matplotlib.pyplot as plt

# 数据生成部分与之前相同
n = 2000  # 样本量大小
X = np.random.rand(n, 2) * 10
Y = np.zeros(n, dtype=int)

for i in range(n):
    if 0 < X[i, 0] < 3 and 0 < X[i, 1] < 3:
        Y[i] = 1
    if 0 < X[i, 0] < 3 and 3.5 < X[i, 1] < 6.5:
        Y[i] = 2
    if 0 < X[i, 0] < 3 and 7 < X[i, 1] < 10:
        Y[i] = 3
    if 3.5 < X[i, 0] < 6.5 and 0 < X[i, 1] < 3:
        Y[i] = 4
    if 3.5 < X[i, 0] < 6.5 and 3.5 < X[i, 1] < 6.5:
        Y[i] = 5
    if 3.5 < X[i, 0] < 6.5 and 7 < X[i, 1] < 10:
        Y[i] = 6
    if 7 < X[i, 0] < 10 and 0 < X[i, 1] < 3:
        Y[i] = 7
    if 7 < X[i, 0] < 10 and 3.5 < X[i, 1] < 6.5:
        Y[i] = 8
    if 7 < X[i, 0] < 10 and 7 < X[i, 1] < 10:
        Y[i] = 9

X = X[Y > 0, :]
Y = Y[Y > 0]
nn = len(Y)

X = np.vstack([X, np.random.rand(n - nn, 2) * 10])
Y = np.concatenate([Y, np.ceil(np.random.rand(n - nn) * 9).astype(int)])

colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#800080', '#008000']
markers = ['o', 's', '^', 'p', '*', 'x', 'D', 'v', 'h']

plt.figure(figsize=(7, 6), facecolor='w')
for i in range(1, 10):
    plt.scatter(X[Y == i, 0], X[Y == i, 1], c=colors[i-1], marker=markers[i-1], label=f'Class {i}', edgecolors='k')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Classification of points')
plt.show()

# 生成测试数据
m = 100  # 测试样本量大小
Xt = np.random.rand(m, 2) * 10
Yt = np.zeros(m, dtype=int)

for i in range(m):
    if 0 < Xt[i, 0] < 3 and 0 < Xt[i, 1] < 3:
        Yt[i] = 1
    if 0 < Xt[i, 0] < 3 and 3.5 < Xt[i, 1] < 6.5:
        Yt[i] = 2
    if 0 < Xt[i, 0] < 3 and 7 < Xt[i, 1] < 10:
        Yt[i] = 3
    if 3.5 < Xt[i, 0] < 6.5 and 0 < Xt[i, 1] < 3:
        Yt[i] = 4
    if 3.5 < Xt[i, 0] < 6.5 and 3.5 < Xt[i, 1] < 6.5:
        Yt[i] = 5
    if 3.5 < Xt[i, 0] < 6.5 and 7 < Xt[i, 1] < 10:
        Yt[i] = 6
    if 7 < Xt[i, 0] < 10 and 0 < Xt[i, 1] < 3:
        Yt[i] = 7
    if 7 < Xt[i, 0] < 10 and 3.5 < Xt[i, 1] < 6.5:
        Yt[i] = 8
    if 7 < Xt[i, 0] < 10 and 7 < Xt[i, 1] < 10:
        Yt[i] = 9

Xt = Xt[Yt > 0, :]
Yt = Yt[Yt > 0]
m = len(Yt)
Ym = np.zeros(m, dtype=int)

# 朴素贝叶斯分类器
class NaiveBayesClassifier:
    def __init__(self):
        self.priors = {}
        self.means = {}
        self.vars = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            X_cls = X[y == cls]
            self.priors[cls] = X_cls.shape[0] / X.shape[0]
            self.means[cls] = X_cls.mean(axis=0)
            self.vars[cls] = X_cls.var(axis=0)

    def predict(self, X):
        posteriors = []
        for x in X:
            class_posteriors = []
            for cls in self.classes:
                prior = np.log(self.priors[cls])
                class_conditional = -0.5 * np.sum(np.log(2 * np.pi * self.vars[cls]))
                class_conditional -= 0.5 * np.sum(((x - self.means[cls]) ** 2) / (self.vars[cls]))
                posterior = prior + class_conditional
                class_posteriors.append(posterior)
            posteriors.append(self.classes[np.argmax(class_posteriors)])
        return np.array(posteriors)

# 训练模型
nb = NaiveBayesClassifier()
nb.fit(X, Y)

# 预测
Ym = nb.predict(Xt)

# 绘制预测结果
plt.figure(figsize=(7, 6), facecolor='w')
for i in range(1, 10):
    plt.scatter(X[Y == i, 0], X[Y == i, 1], c=colors[i-1], marker=markers[i-1], label=f'Class {i}', edgecolors='k')
plt.scatter(Xt[:, 0], Xt[:, 1], c='black', marker='+', label='Test Data', edgecolors='k')
for i in range(1, 10):
    plt.scatter(Xt[Ym == i, 0], Xt[Ym == i, 1], c=colors[i-1], marker='+', facecolors='none', edgecolors=colors[i-1], label=f'Predicted Class {i}')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Classification of points')
plt.show()

# 计算错误率
error_rate = np.mean(Ym != Yt)
print(f'错误率: {error_rate * 100:.2f}%')
