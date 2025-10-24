# 🧠 Deep Learning Experiment — Fashion-MNIST Classification

📘 **Project Overview**  
This folder contains the code, figures, and reports for reproducing and extending **deep learning experiments** based on the *Fashion-MNIST* dataset from the book *Dive into Deep Learning (D2L)*.  
The project focuses on implementing Softmax Regression and Multi-Layer Perceptron (MLP) models, evaluating their performance under different architectures, activation functions, and training settings.

---

## ⚙️ Folder Structure

```plaintext
.
├── 1.png
├── 2.png
├── 3.png
├── 4.png
├── 5.png
├── 6.png
├── 7.png
├── 8.png
├── exp1.py
```

---

## 🚀 Features

- **Softmax Regression (From Scratch)** — Implements a custom softmax classifier and cross-entropy loss.  
- **Multi-Layer Perceptron (MLP)** — Experiments with different network depths, neuron counts, and activations.  
- **Visualization & Evaluation** — Includes PNG figures (`1.png–8.png`) showing accuracy, loss, and prediction results.  
- **Performance Tuning** — Compares different learning rates, epochs, and weight initialization strategies.  
- **Comprehensive Report** — Summarizes experiment setup, results, and analysis in a detailed `.docx` document.  

---

## 🧩 Code Overview

Core logic implemented in [`exp1.py`](./exp1.py):

- **Data Loading:** Uses `torchvision.datasets.FashionMNIST` with preprocessing and batching.  
- **Softmax Implementation:** Built from basic tensor operations with manual backpropagation and gradient updates.  
- **Accuracy Metrics:** Custom evaluator integrated with visualization via an `Animator` class.  
- **MLP Experiments:**  
  - Hidden layers: `256`, `128-64`, `256-256-128`  
  - Activation functions: `ReLU`, `Sigmoid`  
  - Learning rates: `0.1`, `0.3`  
  - Epochs: 10–30  
- **Optimized Model:** Achieves high accuracy with fine-tuned hyperparameters and custom initialization.  

---

## 🧠 Tech Stack

- **Language:** Python 3.10+  
- **Frameworks:** PyTorch, D2L (Dive into Deep Learning)  
- **Visualization:** Matplotlib  


---

## 🚀 Getting Started

1. **Clone the Repository**

   ```bash
   git clone https://github.com/ali-rose/Bachelor_project.git
   cd Bachelor_project/Deeplearning
   ```

2. **Install Dependencies**

   ```bash
   pip install torch torchvision matplotlib d2l
   ```

3. **Run the Experiment**

   ```bash
   python exp1.py
   ```

4. **View Results**

   - Console logs show training progress and accuracy.  
   - Figures (loss curves, predictions) are saved as `1.png–8.png`.  
   - Additional outputs available in the notebook and report.

---

## 🔗 Original Reference

This project is inspired by the open-source textbook:  
👉 [Dive into Deep Learning (D2L.ai)](https://d2l.ai)

---

## 📄 References

- *Fashion-MNIST Dataset*: Zalando Research  
- *D2L Official Repository*: [https://github.com/d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en)  
- *PyTorch Documentation*: [https://pytorch.org](https://pytorch.org)

---

## 🧩 Disclaimer

This project is for **educational and research purposes only**.  
All copyrights belong to their respective authors.

---

## 👨‍💻 Author

**Ailixiaer Ailika**  
Bachelor Thesis Project — *Deep Learning Fundamentals*  
📍 University Project Repository (Non-Commercial Use)

---

## 🪪 License

Released under the **MIT License**.  
Feel free to use, modify, and distribute with proper attribution.
