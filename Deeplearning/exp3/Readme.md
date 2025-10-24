# 🧠 Deep Learning Experiment — Convolutional Neural Network Architecture Optimization (Lab 3)

📘 **Project Overview**  
This folder contains the source code, Jupyter notebook, and all result images for the **third deep learning experiment**, which investigates how **CNN architecture modifications**, **training epochs**, **kernel sizes**, **layer depth**, and **optimizers** affect model performance on the *Fashion-MNIST* dataset.  
The experiment further explores **modular CNN designs** such as *VGG*, *NIN*, and *Inception*.

---

## ⚙️ Folder Structure

```plaintext
.
├── lab3.ipynb                                        # Interactive training notebook  
├── 要求1，十次训练.png                                # 10-epoch training  
├── 要求1、五十次训练.png                              # 50-epoch training  
├── 要求2.png                                          # Modified convolution kernel test  
├── 要求3.png                                          # Pooling layer comparison  
├── 要求4、窗口提升为8.png                             # Pooling window = 8  
├── 要求4、窗口降低为3.png                             # Pooling window = 3  
├── 要求5、第一层输出增加为12.png                       # First conv layer output = 12  
├── 要求5、第一层输出降为3、第二层输出降为4.png         # Reduced conv layer outputs  
├── 要求6、增加一层卷积.png                            # Added convolutional layer  
├── 要求6、减少一层卷积.png                            # Removed convolutional layer  
├── 要求7、改为最大汇聚层.png                          # Switched to MaxPooling layer  
├── 要求8、使用Adam优化器，学习率为0.9.png             # Adam optimizer (lr = 0.9)  
├── 要求8、使用Adam优化器，学习率为1.5.png             # Adam optimizer (lr = 1.5)  
├── 附加要求1、第一层换位VGG模块.png                    # VGG module replacement  
├── 附件要求2、第一层换为NIN模块.png                    # NIN module replacement  
└── 附件要求3、第一层换位googlenet的inception模块.png    # Inception module replacement
```

---

## 🚀 Features

- 🧩 **CNN Architecture Comparison**  
  - Adjusted convolutional depth, kernel sizes, and pooling strategies.  
  - Explored effects of different activation and optimization settings.

- ⚙️ **Advanced Module Integration**  
  - Integrated **VGG**, **NIN**, and **Inception** modules into the base CNN.  
  - Compared performance and feature map dimensionality.

- 📈 **Optimizer Evaluation**  
  - Compared **SGD** and **Adam** optimizers at various learning rates (0.9 / 1.5).

- 🎨 **Visualization Outputs**  
  - Accuracy and loss evolution saved as `.png` images for each experiment condition.

---

## 🧩 Code Overview

Core code from [`lab3.ipynb`](./lab3.ipynb):

```python
import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F

# -------------------- Inception Module --------------------
class Inception(nn.Module):
    """Inception block combining multiple convolutional paths."""
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1: single 1×1 conv
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # Path 2: 1×1 conv → 3×3 conv
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # Path 3: 1×1 conv → 5×5 conv
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # Path 4: 3×3 max-pool → 1×1 conv
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # Concatenate along channel dimension
        return torch.cat((p1, p2, p3, p4), dim=1)

# -------------------- Model Definition --------------------
net = nn.Sequential(
    Inception(1, 6, (3, 6), (3, 6), 6),             # Replace first conv with Inception
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(24, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)

# -------------------- Forward Shape Check --------------------
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:', X.shape)
```

### 🧮 Training and Evaluation

```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Evaluate model accuracy on GPU."""
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# -------------------- Training Function --------------------
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train model with GPU support."""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {l:.3f}, train acc {metric[1]/metric[2]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')

lr, num_epochs = 0.9, 50
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

---

## 🧠 Tech Stack

- **Language:** Python 3.10+  
- **Frameworks:** PyTorch 2.x, D2L (PyTorch Edition)  
- **Dataset:** Fashion-MNIST (via `torchvision.datasets`)  
- **Visualization:** Matplotlib for real-time loss/accuracy plots  
- **Environment:** Jupyter Notebook (`lab3.ipynb`)

---

## 🚀 Getting Started

1. **Clone the Repository**

   ```bash
   git clone https://github.com/ali-rose/Bachelor_project.git
   cd Bachelor_project/Deeplearning/lab3
   ```

2. **Install Dependencies**

   ```bash
   pip install torch torchvision matplotlib d2l
   ```

3. **Run the Experiment**

   ```bash
   python lab3.ipnyb
   # or open lab3.ipynb for step-by-step execution
   ```

4. **View Results**

   - Training curves and accuracy comparisons saved as `.png` files.  
   - Refer to the `.docx` report for detailed analysis and discussion.

---

## 🔗 Original Reference

This experiment builds upon materials from:  
👉 [Dive into Deep Learning (D2L.ai)](https://d2l.ai)

---

## 📄 References

- *PyTorch Official Documentation*: [https://pytorch.org](https://pytorch.org)  
- *D2L Official Repository*: [https://github.com/d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en)  
- *GoogLeNet / Inception Paper*: *Szegedy et al., CVPR 2015*  

---

## 🧩 Disclaimer

This project is for **educational and research purposes only**.  
All intellectual property rights belong to their respective authors.

---

## 👨‍💻 Author

**Ailixiaer Ailika**  
Bachelor Thesis Project — *Deep Learning Fundamentals (Lab 3)*  
📍 University Project Repository (Non-Commercial Use)

---

## 🪪 License

Released under the **MIT License**.  
Free to use, modify, and redistribute with proper attribution.
