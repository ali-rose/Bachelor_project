import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# 数据生成
n = 2000  # 样本量大小
X = np.random.rand(n, 2) * 10  # 生成2n * 2的数据矩阵，每一行表示一个数据点，第一列表示x轴坐标，第二列表示y轴坐标
Y = np.zeros(n)  # 类别标签

for i in range(n):
    if (0 < X[i, 0] < 3 and (0 < X[i, 1] < 3 or 3.5 < X[i, 1] < 6.5 or 7 < X[i, 1] < 10)) or \
       (3.5 < X[i, 0] < 6.5 and (0 < X[i, 1] < 3 or 3.5 < X[i, 1] < 6.5 or 7 < X[i, 1] < 10)) or \
       (7 < X[i, 0] < 10 and (0 < X[i, 1] < 3 or 3.5 < X[i, 1] < 6.5 or 7 < X[i, 1] < 10)):
        Y[i] = 1

X = X[Y > 0]  # 只保留Y>0的数据点
Y = Y[Y > 0]
n = len(Y)  # 更新去除白色间隔后的点的个数

plt.figure(1, figsize=(7, 6), facecolor='w')
plt.scatter(X[:, 0], X[:, 1], c='k', s=10, linewidths=1)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Data Points')
plt.show()

# K-means算法 - 尝试不同的距离度量函数
K = 9  # 中心点个数

def kmeans_custom(X, K, distance_metric='euclidean'):
    # 初始化随机中心
    meanpoint = X[np.random.choice(X.shape[0], K, replace=False)]
    for iteration in range(100):  # 设置迭代次数
        # 计算每个点到所有中心点的距离
        labels, _ = pairwise_distances_argmin_min(X, meanpoint, metric=distance_metric)
        new_meanpoint = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        if np.all(meanpoint == new_meanpoint):
            break
        meanpoint = new_meanpoint
    return labels, meanpoint

# 使用不同的距离度量
distance_metrics = ['euclidean', 'manhattan']
for metric in distance_metrics:
    Ym, meanpoint = kmeans_custom(X, K, distance_metric=metric)

    # 画出聚类结果及中心点
    plt.figure(figsize=(7, 6), facecolor='w')
    colors = ['r', 'k', 'b', 'g', 'm', 'c', 'b', 'r', 'k']
    markers = ['o', 'o', 'o', '*', '*', '*', '+', '+', '+']

    for k in range(K):
        plt.scatter(X[Ym == k, 0], X[Ym == k, 1], c=colors[k], marker=markers[k], s=10, linewidths=1)

    plt.scatter(meanpoint[:, 0], meanpoint[:, 1], c='m', marker='s', s=100, facecolors='m')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title(f'K-means Clustering Results with {metric} distance')
    plt.show()

