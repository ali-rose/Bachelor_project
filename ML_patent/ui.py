import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox

from xgboost import XGBRegressor

class MLModelUI(QWidget):
    def __init__(self, models):
        super().__init__()
        self.models = models  # 一个字典，键是模型的名称，值是模型实例
        self.setWindowTitle('机器学习模型预测工具')
        self.setGeometry(100, 100, 600, 300)  # 增大窗口大小

        self.initUI()

    def initUI(self):
        vbox = QVBoxLayout()

        # 创建下拉列表以选择模型
        self.model_selector = QComboBox()
        self.model_selector.addItems(models.keys())
        vbox.addWidget(self.model_selector)

        # 创建输入框和标签
        labels = ['温度:', '湿度:', '风:', '排放量:']
        self.input_lineedits = []
        for label_text in labels:
            label = QLabel(label_text)
            lineedit = QLineEdit()
            hbox = QHBoxLayout()
            hbox.addWidget(label)
            hbox.addWidget(lineedit)
            self.input_lineedits.append(lineedit)
            vbox.addLayout(hbox)

        # 创建预测按钮和输出标签
        self.predict_button = QPushButton('预测')
        self.predict_button.clicked.connect(self.predict)
        self.clear_button = QPushButton('清零')  # 添加清零按钮
        self.clear_button.clicked.connect(self.clearInputs)  # 绑定清零功能

        # 设置按钮样式
        self.predict_button.setStyleSheet("QPushButton { background-color: #A3C1DA; color: white; }")
        self.clear_button.setStyleSheet("QPushButton { background-color: #C1A3DA; color: white; }")

        self.output_label = QLabel('预测结果:')
        vbox.addWidget(self.predict_button)
        vbox.addWidget(self.clear_button)  # 将清零按钮添加到布局中
        vbox.addWidget(self.output_label)

        self.setLayout(vbox)

    def predict(self):
        # 根据下拉列表的选择确定使用哪个模型
        selected_model_name = self.model_selector.currentText()
        model = self.models[selected_model_name]

        # 获取输入值
        input_values = [float(lineedit.text()) for lineedit in self.input_lineedits]
        prediction = model.predict([input_values])
        self.output_label.setText(f'预测结果: {prediction[0]}')

    def clearInputs(self):
        # 清除所有输入框
        for lineedit in self.input_lineedits:
            lineedit.clear()
        self.output_label.setText('预测结果:')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 加载您的机器学习模型
    model_no2 = XGBRegressor()
    model_no2.load_model('E:/XGboost/xgboost_model_oxide.bin')  # 加载氮氧化物预测模型
    model_ozone = XGBRegressor()
    model_ozone.load_model('E:/XGboost/xgboost_model_ozone.bin')  # 加载臭氧预测模型
    models = {'氮氧化物': model_no2, '臭氧': model_ozone}
    window = MLModelUI(models)
    window.show()
    sys.exit(app.exec_())
