import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min

# 数据生成
n = 2000
X = np.random.rand(n, 2) * 10
Y = np.zeros(n)

for i in range(n):
    if 0 < X[i, 0] < 6 and 0 < X[i, 1] < 6:
        Y[i] = 1
    if 7 < X[i, 0] < 10 and 0 < X[i, 1] < 3:
        Y[i] = 2
    if 7 < X[i, 0] < 10 and 3 < X[i, 1] < 6:
        Y[i] = 3
    if 0 < X[i, 0] < 3 and 7 < X[i, 1] < 10:
        Y[i] = 4
    if 3 < X[i, 0] < 6 and 7 < X[i, 1] < 10:
        Y[i] = 5
    if 7 < X[i, 0] < 10 and 7 < X[i, 1] < 10:
        Y[i] = 6

X = X[Y > 0]
Y = Y[Y > 0]
n = len(Y)

# 可视化数据
plt.figure(1, figsize=(7, 6))
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c='r', marker='o', label='Class 1')
plt.scatter(X[Y == 2, 0], X[Y == 2, 1], c='k', marker='o', label='Class 2')
plt.scatter(X[Y == 3, 0], X[Y == 3, 1], c='b', marker='o', label='Class 3')
plt.scatter(X[Y == 4, 0], X[Y == 4, 1], c='g', marker='*', label='Class 4')
plt.scatter(X[Y == 5, 0], X[Y == 5, 1], c='m', marker='*', label='Class 5')
plt.scatter(X[Y == 6, 0], X[Y == 6, 1], c='c', marker='*', label='Class 6')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend()
plt.show()

# K-means算法实现
K = 6
max_iter = 100
tolerance = 1e-4

def kmeans_custom_distance(X, K, max_iter=100, tolerance=1e-4, metric='euclidean'):
    # 随机初始化中心点
    meanpoint = np.random.rand(K, 2) * 10
    Ym = np.zeros(n)

    for _ in range(max_iter):
        # 计算每个点到各中心点的距离，并分配到最近的中心点
        Ym, _ = pairwise_distances_argmin_min(X, meanpoint, metric=metric)

        # 计算新的中心点
        new_meanpoint = np.array([X[Ym == k].mean(axis=0) for k in range(K)])

        # 判断中心点是否收敛
        if np.linalg.norm(new_meanpoint - meanpoint) < tolerance:
            break

        meanpoint = new_meanpoint

    return Ym, meanpoint

# 使用不同距离度量进行聚类
metrics = ['euclidean', 'manhattan']
for metric in metrics:
    Ym, meanpoint = kmeans_custom_distance(X, K, metric=metric)

    # 可视化聚类结果及中心点
    plt.figure(figsize=(7, 6))
    plt.scatter(X[Ym == 0, 0], X[Ym == 0, 1], c='r', marker='o', label='Cluster 1')
    plt.scatter(X[Ym == 1, 0], X[Ym == 1, 1], c='k', marker='o', label='Cluster 2')
    plt.scatter(X[Ym == 2, 0], X[Ym == 2, 1], c='b', marker='o', label='Cluster 3')
    plt.scatter(X[Ym == 3, 0], X[Ym == 3, 1], c='g', marker='*', label='Cluster 4')
    plt.scatter(X[Ym == 4, 0], X[Ym == 4, 1], c='m', marker='*', label='Cluster 5')
    plt.scatter(X[Ym == 5, 0], X[Ym == 5, 1], c='c', marker='*', label='Cluster 6')
    plt.scatter(meanpoint[:, 0], meanpoint[:, 1], c='m', marker='s', edgecolors='k', s=100, label='Centroids')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.legend()
    plt.title(f'K-means Clustering with {metric.capitalize()} Distance')
    plt.show()