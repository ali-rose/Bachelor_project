import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
import warnings
warnings.filterwarnings('ignore')

#1导入数据
datasets = pd.read_excel('E:/XGboost/finalmulti(Ozone).xlsx')
dataset = datasets.iloc[0:6721,0:5].values



X=datasets[['Temperature','Humidity','Wind','Emission']]
Y=datasets['Ozone']

#划分数据集和训练集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


#4搭建预测模型
xgb = XGBRegressor(booster='gbtree',max_depth=3, learning_rate=0.2,reg_alpha=0.01, n_estimators=2000, gamma=0.4, min_child_weight=1)
#xgb = XGBRegressor(booster='gbtree',max_depth=40, learning_rate=0.2,reg_alpha=0.01, n_estimators=2000, gamma=0.1, min_child_weight=1)
xgb.fit(X_train,Y_train)


pre = xgb.predict(X_test)
#重新排列行索引
arr_1 = np.arange(0, len(Y_test))
# 将新的索引赋给数组
Y_test = np.column_stack((arr_1, Y_test))
#删掉第一列索引列防止发生错误
Y_test = np.delete(Y_test, 0, axis=1)
#扁平化
Y_test = Y_test.flatten()
pre = pre.flatten()
#将最终结果的索引相互对齐
df = pd.DataFrame({'Y_test':Y_test,'pre':pre})


#5指标重要性可视化
importance = xgb.feature_importances_
plt.figure(1)
plt.barh(y = range(importance.shape[0]),  #指定条形图y轴的刻度值
         width = importance,  #指定条形图x轴的数值
         tick_label =range(importance.shape[0]),  #指定条形图y轴的刻度标签
         color = 'orangered',  #指定条形图的填充色
         )
plt.title('Feature importances of XGBoost')

#6计算评价指标
print(' MAE : ', mae(Y_test, pre))
print(' MAPE : ',mape(Y_test, pre))
print(' RMSE : ', np.sqrt(mse(Y_test, pre)))

#7结果可视化
plt.figure(2)
#plt.scatter(df.index, df['pre'], color='red', label='predict', marker='o')
#plt.scatter(df.index, df['Y_test'], color='blue', label='true', marker='x')
plt.plot(df.pre, color='red',label='predict')
plt.plot(Y_test, color='blue',label='true')
plt.title('Result visualization')
plt.legend()
plt.show()


# 保存模型到文件
xgb.save_model('E:/XGboost/xgboost_model_ozone.bin')


# 加载模型
# loaded_model = joblib.load('xgboost_model.pkl')

# 使用加载的模型进行预测
# predictions = loaded_model.predict(X_test)