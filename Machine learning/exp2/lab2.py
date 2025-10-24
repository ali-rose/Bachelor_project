import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(0)
class1 = np.random.randn(100, 2) + [1, 1]
class2 = np.random.randn(100, 2) * 2 + [5, 5]

X = np.vstack([class1, class2])
Y = np.array([1] * 100 + [2] * 100)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def chebyshev_distance(x1, x2):
    return np.max(np.abs(x1 - x2))

def k_nearest_neighbors(X_train, y_train, x_test, k):
    distances = []
    for x_train in X_train:
        distances.append(euclidean_distance(x_train, x_test))
    distances = np.array(distances)
    # 获取距离最近的k个点的索引
    k_indices = np.argsort(distances)[:k]
    # 找出这些点的类别
    k_nearest_labels = y_train[k_indices]
    # 返回出现次数最多的类别
    (values, counts) = np.unique(k_nearest_labels, return_counts=True)
    return values[np.argmax(counts)]

# 可视化数据
plt.scatter(class1[:, 0], class1[:, 1], color='red', label='Class 1')
plt.scatter(class2[:, 0], class2[:, 1], color='blue', label='Class 2')
plt.legend()
plt.show()

# 不同K值的影响
for k in [1, 3, 5, 10,15,20,30,100]:
    predictions = []
    for x_test in X:
        label = k_nearest_neighbors(X, Y, x_test, k)
        predictions.append(label)

    error_rate = np.mean(predictions != Y)
    print(f'K={k}: Error rate = {error_rate:.2f}')
