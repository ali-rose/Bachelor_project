# 🩺 Pneumonia Image Classification — LeNet & ResNet Experiments

📘 **Project Overview**  
This repository contains two implementations of pneumonia image classification models based on deep convolutional neural networks:  
- **LeNet-5 (TensorFlow/Keras)**  
- **Modified ResNet (PyTorch)**  

Both models were trained and evaluated on the **Chest X-Ray Pneumonia dataset**, aiming to distinguish between **normal** and **pneumonia-infected** chest X-ray images.  
The experiments demonstrate differences in training strategies, frameworks, and model depth when applied to a medical imaging binary classification task.

---

## ⚙️ Folder Structure

```plaintext
.
├── main_Lenet.py                     # LeNet model implementation using Keras
├── main_ResNet.ipynb                    # Modified ResNet model using PyTorch
└── README.md                         # This documentation
```

---

## 🚀 Features

- 🧩 **Two Framework Implementations**
  - **LeNet-5 (TensorFlow)** — A classic shallow convolutional network.
  - **ResNet (PyTorch)** — A modern deep residual architecture adapted for binary classification.

- 🩻 **Dataset**
  - Source: Chest X-Ray Pneumonia dataset (Kaggle).  
  - Categories: `NORMAL` and `PNEUMONIA`.  
  - Images resized to **200×200 (LeNet)** or **96×96 (ResNet)**.  

- 📈 **Training Enhancements**
  - Data augmentation for robustness.  
  - Validation monitoring and visualization.  
  - Use of GPU acceleration when available.

---

## 🧩 Experiment 1 — LeNet (Keras / TensorFlow)

### 🔹 Architecture Overview
```plaintext
Input (200×200×3)
↓
Conv2D → ReLU → MaxPooling
↓
Conv2D → ReLU → MaxPooling
↓
Flatten → Dense(120) → Dense(84)
↓
Dense(2) → Softmax
```

### 🔹 Training Setup
| Parameter | Value |
|------------|--------|
| Optimizer | Adam |
| Loss Function | Categorical CrossEntropy |
| Batch Size | 32 |
| Epochs | 10 |
| Image Size | 200×200×3 |
| Dataset Split | 80% train / 20% test |

### 🔹 Sample Training Log
```
Epoch 1/10  - loss: 0.4466 - acc: 0.8146 - val_acc: 0.6875  
Epoch 5/10  - loss: 0.1138 - acc: 0.9576 - val_acc: 0.8125  
Epoch 10/10 - loss: 0.0186 - acc: 0.9948 - val_acc: 0.7500  
Test Accuracy: 0.7484
```

### 🔹 Key Observations
- Model quickly converges with high training accuracy (~99%).  
- Validation accuracy fluctuates (~75–88%), indicating slight overfitting.  
- Suitable for smaller datasets and faster prototyping.

---

## 🧩 Experiment 2 — Modified ResNet (PyTorch)

### 🔹 Model Design
- **BinaryResidual Block:** Two 3×3 convolution layers with skip connections.  
- **Adaptive Average Pooling** to flatten spatial dimensions.  
- **Final Linear Layer** outputs two classes.  

### 🔹 Network Structure
```plaintext
Input (3×96×96)
↓
7×7 Conv → BN → ReLU → MaxPool
↓
Residual Blocks (b2–b5)
↓
AdaptiveAvgPool2D(1×1)
↓
Flatten → Linear(512→2)
```

### 🔹 Training Setup
| Parameter | Value |
|------------|--------|
| Optimizer | SGD |
| Learning Rate | 0.05 |
| Epochs | 10 |
| Batch Size | 32 |
| Loss Function | CrossEntropy |
| Validation Split | 80% train / 20% val |

### 🔹 Training Output Example
```
Epoch 1/10: Loss=0.5734, Train Acc=0.81, Val Acc=0.78  
Epoch 5/10: Loss=0.1621, Train Acc=0.94, Val Acc=0.88  
Epoch 10/10: Loss=0.0945, Train Acc=0.97, Val Acc=0.90
```

### 🔹 Results
| Metric | LeNet | ResNet |
|---------|--------|--------|
| Test Accuracy | 74.8% | 90.2% |
| Model Depth | 5 Conv Layers | 20+ Layers |
| Framework | TensorFlow | PyTorch |
| Training Time | Short | Longer but more stable |

### 🔹 Insights
- ResNet achieves **higher accuracy and generalization**, benefiting from residual learning.  
- LeNet remains **computationally lightweight** and suitable for smaller datasets or edge devices.  

---

## 🧠 Comparative Summary

| Aspect | LeNet (Keras) | ResNet (PyTorch) |
|--------|----------------|------------------|
| Framework | TensorFlow 2.x | PyTorch |
| Dataset | 200×200 RGB | 96×96 RGB |
| Architecture | Shallow CNN | Deep Residual CNN |
| Accuracy | ~75% | ~90% |
| Suitability | Lightweight models | High-accuracy research tasks |

---

## ⚙️ Run the Code

### ▶ LeNet (TensorFlow)
```bash
pip install tensorflow keras numpy matplotlib
python main_Lenet.py
```

### ▶ ResNet (PyTorch)
```bash
pip install torch torchvision matplotlib scikit-learn
python main_ResNet.py
```

---

## 📄 References

- LeCun, Y. et al. (1998). *Gradient-Based Learning Applied to Document Recognition.* Proceedings of the IEEE.  
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR.  
- Chest X-Ray Pneumonia Dataset — [Kaggle Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## 👨‍💻 Author

**Ailixiaer Ailika**  
Bachelor Project — *Artificial Intelligence Techniques (Experiments 5 & 6)*  
📍 University Project Repository (Non-Commercial Educational Use)

---

## 🪪 License

Released under the **MIT License**.  
You are free to use, modify, and distribute this code with proper citation.
