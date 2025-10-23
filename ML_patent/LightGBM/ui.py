import sys
import lightgbm as lgb
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton

class MLModelUI(QWidget):
    def __init__(self, model):
        super().__init__()
        self.model = model  # 将模型作为实例变量存储
        self.setWindowTitle('机器学习模型UI')
        self.setGeometry(100, 100, 400, 200)

        vbox = QVBoxLayout()  # 在这里初始化QVBoxLayout

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
            vbox.addLayout(hbox)  # 在这里将hbox添加到vbox中

        # 创建预测按钮和输出标签
        self.predict_button = QPushButton('预测')
        self.predict_button.clicked.connect(self.predict)
        self.output_label = QLabel('Nitrogen Dioxide:')
        vbox.addWidget(self.predict_button)
        vbox.addWidget(self.output_label)

        self.setLayout(vbox)  # 只设置一次布局

    def predict(self):
        # 获取输入值
        input_values = [float(lineedit.text()) for lineedit in self.input_lineedits]
        prediction = self.model.predict([input_values])  # 使用存储的模型进行预测
        self.output_label.setText(f'Nitrogen Dioxide: {prediction[0]}')  # 调整以适应模型的输出格式

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 在这里加载你的机器学习模型
    bst = lgb.Booster(model_file='E:/LightGBM/lightgbm_model2.bin')
    model = bst  # 假设bst是您训练好的模型对象
    window = MLModelUI(model)  # 将模型作为参数传递
    window.show()
    sys.exit(app.exec_())
