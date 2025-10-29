import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def plot_svm_with_different_C(C_values, X, Y, center1, center2):
    plt.figure(figsize=(20, 16))

    for i, C in enumerate(C_values):
        # 训练SVM模型
        clf = SVC(kernel='linear', C=C)
        clf.fit(X, Y)

        # 分类器参数
        w = clf.coef_[0]
        b = clf.intercept_[0]

        # 分类器可视化
        x1 = np.linspace(-2, 7, 400)
        y1 = (-b - w[0] * x1) / w[1]  # 分类界面
        y2 = (1 - b - w[0] * x1) / w[1]  # 上间隔边界
        y3 = (-1 - b - w[0] * x1) / w[1]  # 下间隔边界

        support_vectors = clf.support_vectors_
        support_vector_labels = Y[clf.support_]
        calculated_bs = support_vector_labels - np.dot(support_vectors, w)

        # 在间隔边界上的支持向量
        epsilon = 1e-5
        on_boundary = np.abs(np.abs(np.dot(support_vectors, w) + b) - 1) < epsilon

        plt.subplot(2, 2, i + 1)
        plt.scatter(X[:n, 0], X[:n, 1], color='g', marker='o', s=100, label='class 1')  # 画第一类数据点
        plt.scatter(X[n:, 0], X[n:, 1], color='b', marker='*', s=100, label='class 2')  # 画第二类数据点
        plt.plot(x1, y1, 'k-', linewidth=2, label='classification surface')  # 画分类界面
        plt.plot(x1, y2, 'k--', linewidth=1, label='boundary')  # 画上间隔边界
        plt.plot(x1, y3, 'k--', linewidth=1)  # 画下间隔边界

        # 画支持向量
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100,
                    facecolors='none', edgecolors='r', label='support vectors')
        # 画在间隔边界上的支持向量
        plt.scatter(support_vectors[on_boundary, 0], support_vectors[on_boundary, 1], s=100,
                    facecolors='purple', edgecolors='purple', label='support vectors on boundary')

        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.legend()
        plt.title(f'SVM classification with C={C}')

    plt.tight_layout()
    plt.show()


# 数据生成
n = 100  # 样本量大小
center1 = np.array([1, 1])  # 第一类数据中心
center2 = np.array([3, 3])  # 第二类数据中心; 线性不可分数据，可以改为 [3, 3]
X = np.zeros((2 * n, 2))  # 2n * 2 的数据矩阵，每一行表示一个数据点，第一列表示 x 轴坐标，第二列表示 y 轴坐标
Y = np.zeros(2 * n)  # 类别标签

# 生成第一类数据
X[:n, :] = center1 + np.random.randn(n, 2)
# 生成第二类数据
X[n:, :] = center2 + np.random.randn(n, 2)
Y[:n] = 1
Y[n:] = -1

# 调整不同的C值
C_values = [0.01, 0.1, 1, 10000]

# 可视化不同C值下的分类结果
plot_svm_with_different_C(C_values, X, Y, center1, center2)
