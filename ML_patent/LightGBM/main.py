import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm



# 从CSV文件中加载数据集
data = pd.read_csv('氮氧.csv')

# 划分特征和标签
X = data.drop(columns=['Monitoring Station'])
y = data['Monitoring Station']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建LightGBM数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# 设置参数
params = {
    'objective': 'regression',
    'metric': 'rmse',
}

# 训练模型
num_round = 100
callbacks = [
    lgb.early_stopping(stopping_rounds=10),
    lgb.log_evaluation(period=0)  # 设置 period 为 0 来禁用日志输出
]
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data], callbacks=callbacks)

# 在测试集上进行预测
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

# 计算均方根误差
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"Root Mean Squared Error: {rmse}")




pre = bst.predict(X_test)
# 重新排列行索引
arr_1 = np.arange(0, len(y_test))
# 将新的索引赋给数组
Y_test = np.column_stack((arr_1, y_test))
# 删掉第一列索引列防止发生错误
Y_test = np.delete(Y_test, 0, axis=1)
# 扁平化
Y_test = Y_test.flatten()
pre = pre.flatten()
# 将最终结果的索引相互对齐
df = pd.DataFrame({'Y_test': Y_test, 'pre': pre})


def plot(y_tru, y_pre, teacher_len):
    import matplotlib.pyplot as plt
    # 示例的预测值和真实值


predictions = y_pred
ground_truth = Y_test
# 创建 x 轴坐标，可以是简单的范围
x = range(len(predictions))
# 使用Matplotlib绘制折线图
plt.figure(figsize=(8, 4))  # 设置图像大小

# 绘制真实值的折线
plt.plot(ground_truth, label='Ground Truth')
# 绘制预测值的折线
plt.plot(predictions, label='Prediction')
# 添加图例
plt.legend()

# 添加标签
plt.xlabel('Data')
plt.ylabel('Ozone')

# 显示网格
plt.grid(True)

# 显示图像
plt.title('Pre vs. Tru')
plt.title('Pre vs. Tru')

plt.show()


def mape(y_true, y_pred):
    """
    计算平均绝对百分比误差（MAPE）
    参数:
    y_true -- 真实值
    y_pred -- 预测值
    返回:
    mape -- MAPE值
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true != 0)
    return np.mean(np.abs((y_true[non_zero_index] - y_pred[non_zero_index]) / y_true[non_zero_index])) * 100

#计算评价指标
Y_pred = bst.predict(X_test)  # 计算输出变量的预测值
print("平均绝对值误差MAE:%0.3f"%sm.mean_absolute_error(y_test, y_pred))
print("均方误差MSE:%0.3f"%sm.mean_squared_error(y_test, y_pred,squared=True))
print("均方根误差RMSE:%0.3f"%sm.mean_squared_error(y_test,y_pred,squared=False))
print("R2决定系数:%0.3f"%sm.r2_score(y_test, y_pred))
print('平均绝对百分误差MAPE : ',mape(y_test, y_pred))
# accuracy=accuracy_score(Y_test,Y_pred)

# 如果你希望以二进制格式保存模型（这样可以更高效地加载模型），你可以这样做：
bst.save_model('E:/LightGBM/lightgbm_model2.bin')

# 如果模型是以二进制格式保存的，同样的方法也适用
#bst = lgb.Booster(model_file='E:/LightGBM/lightgbm_model.bin')