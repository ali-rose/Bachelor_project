# 🧠 Deep Learning Experiment — Learning Rate Schedulers & Initialization Methods

📘 **Project Overview**  
This folder contains the code, reports, and output figures for the **second deep learning experiment**, focusing on the effects of **weight initialization** (Gaussian and Xavier) and **learning rate scheduling** on model convergence and performance.  
The experiment builds on *Fashion-MNIST* using fully connected neural networks implemented in **PyTorch** and **D2L (Dive into Deep Learning)**.

---

## ⚙️ Folder Structure

```plaintext
├── lab2.py                                        # Core experiment code (PyTorch)  
├── 报告中附加题图像汇总/                          # Additional visual outputs used in report  
├── 要求一高斯初始化图像/                          # Gaussian initialization - Experiment 1 results  
├── 要求一Xavier初始化图像/                        # Xavier initialization - Experiment 1 results  
├── 要求二高斯初始化图像/                          # Gaussian initialization - Experiment 2 results  
├── 要求二Xavier初始化图像/                        # Xavier initialization - Experiment 2 results  
├── 要求三高斯初始化图像/                          # Gaussian initialization - Experiment 3 results  
└── 要求三Xavier初始化图像/                        # Xavier initialization - Experiment 3 results
```

---

## 🚀 Features

- **Weight Initialization Comparison**
  - Gaussian (Normal) initialization  
  - Xavier (Glorot) initialization  

- **Learning Rate Scheduler Experiments**
  - Fixed learning rate  
  - Exponential decay  
  - Polynomial decay (Multi-Step)  
  - Cosine annealing  
  - Cosine annealing with warm-up  

- **Training Visualization**
  - Dynamic plots for training loss, accuracy, and validation accuracy via the `Animator` class  
  - Figures automatically saved in categorized folders  

- **Optimizers and Evaluation**
  - Implemented **SGD** optimizer with multiple batch size configurations  
  - Custom training loop supporting both PyTorch built-in and custom learning rate schedulers  

---

## 🧩 Code Overview

Key components in [`lab2.py`](./lab2.py):

- **Custom `train()` Function:**  
  Modified D2L’s original function to support both built-in (`lr_scheduler`) and user-defined schedulers.  

- **Network Architecture:**  
  A 3-layer fully connected neural network:  
  ```
  Input (784) → Linear(256) → ReLU → Linear(128) → ReLU → Linear(10)
  ```

- **Initialization Methods:**  
  ```python
  def init_normal(m):
      if type(m) == nn.Linear:
          nn.init.normal_(m.weight, mean=0, std=0.01)
          nn.init.zeros_(m.bias)
  net.apply(init_normal)
  ```
  Additional experiments used Xavier initialization for comparison.

- **Schedulers Implemented:**  
  - `SquareRootScheduler` (custom exponential decay)  
  - `MultiStepLR` (PyTorch built-in polynomial decay)  
  - `CosineScheduler` (custom cosine decay with optional warm-up)  

- **Evaluation Metrics:**  
  - Training and test accuracy  
  - Loss trajectory visualization  
  - Learning rate variation plots for each scheduling strategy  

---

## 🧠 Tech Stack

- **Language:** Python 3.10+  
- **Frameworks:** PyTorch, D2L (Dive into Deep Learning)  
- **Visualization:** Matplotlib  
- **Dataset:** Fashion-MNIST (via `torchvision.datasets`)

---

## 🚀 Getting Started

1. **Clone the Repository**

   ```bash
   git clone https://github.com/ali-rose/Bachelor_project.git
   cd Bachelor_project/Deeplearning/exp2
   ```

2. **Install Dependencies**

   ```bash
   pip install torch torchvision matplotlib d2l
   ```

3. **Run the Experiment**

   ```bash
   python lab2.py
   ```


---

## 🔗 Original Reference

This experiment is adapted from the open-source textbook:  
👉 [Dive into Deep Learning (D2L.ai)](https://d2l.ai)

---

## 📄 References

- *PyTorch Official Documentation*: [https://pytorch.org](https://pytorch.org)  
- *D2L Official Repository*: [https://github.com/d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en)  
- *Fashion-MNIST Dataset*: Zalando Research  

---

## 🧩 Disclaimer

This project is for **educational and research purposes only**.  
All copyrights belong to their respective authors.

---

## 👨‍💻 Author

**Ailixiaer Ailika**  
Bachelor Thesis Project — *Deep Learning Fundamentals (exp 2)*  
📍 University Project Repository (Non-Commercial Use)

---

## 🪪 License

Released under the **MIT License**.  
Free to use, modify, and redistribute with attribution.
