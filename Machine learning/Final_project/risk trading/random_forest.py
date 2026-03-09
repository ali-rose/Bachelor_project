import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier

# 加载数据
train_data = pd.read_csv('train.csv')
pred_data = pd.read_csv('pred.csv')

# 提取特征和标签
X_train = train_data.drop(columns=['ID', 'Label'])
y_train = train_data['Label']

# 对特征进行标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 对预测数据也进行同样的标准化处理
X_pred = pred_data.drop(columns=['ID'])
X_pred_scaled = scaler.transform(X_pred)

# 划分训练集和验证集
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# 训练随机森林模型
rf_model = RandomForestClassifier()
rf_model.fit(X_train_split, y_train_split)

# 在验证集上进行评估
y_val_pred_rf = rf_model.predict(X_val_split)
print("Random Forest Classification Report")
print(classification_report(y_val_split, y_val_pred_rf))
print(confusion_matrix(y_val_split, y_val_pred_rf))

# 在验证集上计算F1分数
f1_rf = f1_score(y_val_split, y_val_pred_rf)
print("Random Forest F1 Score on Validation Set:", f1_rf)

# 在预测数据上进行预测
y_pred_rf = rf_model.predict(X_pred_scaled)

# 将结果保存到新的DataFrame中
pred_results_rf = pd.DataFrame({
    'ID': pred_data['ID'],
    'label': y_pred_rf
})

# 保存预测结果
pred_results_rf.to_csv('random_forest_pred_results.csv', index=False)
