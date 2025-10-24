import numpy as np
import matplotlib.pyplot as plt

# 数据生成
n = 100
X = np.random.rand(n, 2) * 10
Y = np.zeros(n, dtype=int)

# 根据坐标位置分类
def classify_points(x, y):
    if 0 < x < 3:
        if 0 < y < 3:
            return 1
        elif 3.5 < y < 6.5:
            return 2
        elif 7 < y < 10:
            return 3
    elif 3.5 < x < 6.5:
        if 0 < y < 3:
            return 4
        elif 3.5 < y < 6.5:
            return 5
        elif 7 < y < 10:
            return 6
    elif 7 < x < 10:
        if 0 < y < 3:
            return 7
        elif 3.5 < y < 6.5:
            return 8
        elif 7 < y < 10:
            return 9
    return 0

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def chebyshev_distance(x1, x2):
    return np.max(np.abs(x1 - x2))

def k_nearest_neighbors(X_train, y_train, x_test, k):
    distances = []
    for x_train in X_train:
        distances.append(chebyshev_distance(x_train, x_test))
    distances = np.array(distances)
    # 获取距离最近的k个点的索引
    k_indices = np.argsort(distances)[:k]
    # 找出这些点的类别
    k_nearest_labels = y_train[k_indices]
    # 返回出现次数最多的类别
    (values, counts) = np.unique(k_nearest_labels, return_counts=True)
    return values[np.argmax(counts)]


Y = np.array([classify_points(x, y) for x, y in X])

# 移除未被分类的数据点
mask = Y > 0
X = X[mask]
Y = Y[mask]

# 可视化
plt.figure(figsize=(10, 8))
colors = ['r', 'k', 'b', 'g', 'm', 'c', 'b', 'r', 'k']
markers = ['o', 'o', 'o', '*', '*', '*', '+', '+', '+']
for i in range(1, 10):
    plt.scatter(X[Y == i][:, 0], X[Y == i][:, 1], color=colors[i-1], marker=markers[i-1], s=100, label=f'Class {i}')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Classification of points')
plt.legend()
plt.show()

# 生成和分类测试数据
m = 500
Xt = np.random.rand(m, 2) * 10
Yt = np.array([classify_points(x, y) for x, y in Xt])
mask = Yt > 0
Xt = Xt[mask]
Yt = Yt[mask]

# 可视化测试数据
plt.figure(figsize=(10, 8))
for i in range(1, 10):
    plt.scatter(X[Y == i][:, 0], X[Y == i][:, 1], color=colors[i-1], marker=markers[i-1], s=100, label=f'Class {i} Train')
    plt.scatter(Xt[Yt == i][:, 0], Xt[Yt == i][:, 1], color=colors[i-1], marker='s', facecolors='none', s=100, label=f'Class {i} Test')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Training and Test Data Classification')
plt.legend()
plt.show()

k = 3  # 设定k值
Ym = np.zeros(len(Xt), dtype=int)  # 存储预测结果

for i, x_test in enumerate(Xt):
    Ym[i] = k_nearest_neighbors(X, Y, x_test, k)

# 计算错误率
error_rate = np.mean(Ym != Yt)
print(f'Error rate: {error_rate:.2f}')

# 可视化预测结果
plt.figure(figsize=(10, 8))
colors = ['r', 'k', 'b', 'g', 'm', 'c', 'b', 'r', 'k']
markers = ['o', 'o', 'o', '*', '*', '*', '+', '+', '+']
for i in range(1, 10):
    plt.scatter(X[Y == i][:, 0], X[Y == i][:, 1], color=colors[i-1], marker=markers[i-1], s=100, label=f'Class {i} Train')
    plt.scatter(Xt[Ym == i][:, 0], Xt[Ym == i][:, 1], color=colors[i-1], marker='s', facecolors='none', edgecolor=colors[i-1], s=100, label=f'Class {i} Predict')

plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Manual K-NN Classification Prediction')
plt.legend()
plt.show()
