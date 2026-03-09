import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 加载数据
data_path = 'cnews.train.txt'
data = []
with open(data_path, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            data.append(parts)

# 创建DataFrame
df = pd.DataFrame(data, columns=['category', 'text'])

# 显示数据的前几行
print(df.head())

# 检查数据类别分布
category_distribution = df['category'].value_counts()
print(category_distribution)

# 划分数据集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

# 文本向量化
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['text'])
X_val = vectorizer.transform(val_df['text'])
X_test = vectorizer.transform(test_df['text'])

# 标签编码
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['category'])
y_val = label_encoder.transform(val_df['category'])
y_test = label_encoder.transform(test_df['category'])

# 训练逻辑回归模型
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train)

# 验证逻辑回归模型
val_predictions = log_reg_model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
print(f'Logistic Regression Validation Accuracy: {val_accuracy}')
print(classification_report(y_val, val_predictions, target_names=label_encoder.classes_))

# 测试逻辑回归模型
test_predictions = log_reg_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f'Logistic Regression Test Accuracy: {test_accuracy}')
print(classification_report(y_test, test_predictions, target_names=label_encoder.classes_))

# 训练SVM模型
svm_model = SVC()
svm_model.fit(X_train, y_train)

# 验证SVM模型
val_predictions = svm_model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
print(f'SVM Validation Accuracy: {val_accuracy}')
print(classification_report(y_val, val_predictions, target_names=label_encoder.classes_))

# 测试SVM模型
test_predictions = svm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f'SVM Test Accuracy: {test_accuracy}')
print(classification_report(y_test, test_predictions, target_names=label_encoder.classes_))

# 训练朴素贝叶斯模型
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# 验证朴素贝叶斯模型
val_predictions = nb_model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
print(f'Naive Bayes Validation Accuracy: {val_accuracy}')
print(classification_report(y_val, val_predictions, target_names=label_encoder.classes_))

# 测试朴素贝叶斯模型
test_predictions = nb_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f'Naive Bayes Test Accuracy: {test_accuracy}')
print(classification_report(y_test, test_predictions, target_names=label_encoder.classes_))
