# 🧠 Machine Learning Experiment — Perceptron Classification (Lab 1)

📘 **Project Overview**  
This folder contains the source code, results, and experiment report for the **first machine learning experiment**, which implements and analyzes a **Perceptron binary classifier**.  
The experiment explores **model initialization**, **training with gradient updates**, **decision boundary visualization**, and **accuracy evaluation** under different hyperparameter and data-distribution settings.

---

## ⚙️ Folder Structure

```plaintext
.
├── lab1.py                             # Perceptron training and visualization code
├── lab2.py                             # (Used in follow-up experiments)
├── 1.png                               # Classification boundary visualization
├── 2.png                               # 10-epoch training result
├── 3.png                               # 50-epoch training result
├── 4.png                               # 100-epoch training result
├── 5.png                               # Center-near data visualization
├── 6.png                               # Center-far data visualization
```

---

## 🚀 Features

- **Perceptron Model Implementation**  
  - Binary linear classifier trained with **stochastic gradient descent (SGD)**.  
  - Adjustable learning rate and number of epochs.  

- **Visualization & Evaluation**  
  - Plots of decision boundary and data distribution.  
  - Displays classification accuracy and error rate.  

- **Extended Experiments**  
  - Impact of iteration count (10 / 50 / 100 epochs).  
  - Influence of data-center distance (close vs far).  
  - Effect of feature dimensionality (2D → 100D).

---

## 🧩 Code Overview

### 🔹 Model Initialization
```python
w = np.zeros(2)       # Weight vector
b = 0                 # Bias term
learning_rate = 0.1
epochs = 10
```

### 🔹 Training Algorithm
```python
for epoch in range(epochs):
    for i in range(2 * n):
        if Y[i] * (np.dot(w, X[i]) + b) <= 0:
            w += learning_rate * Y[i] * X[i]
            b += learning_rate * Y[i]
```

### 🔹 Decision Boundary Plot
```python
x_vals = np.linspace(X[:,0].min(), X[:,0].max(), 300)
y_vals = -(b + w[0] * x_vals) / w[1]
plt.plot(x_vals, y_vals, 'k--', label='classification boundary')
```

### 🔹 Testing Accuracy
```python
predictions = np.sign(np.dot(Xt, w) + b)
error_rate = np.mean(predictions != Yt)
accuracy = 1 - error_rate
print("Accuracy:", accuracy)
```

---

## 📊 Experimental Findings

| Setting | Description | Accuracy |
|----------|--------------|-----------|
| 10 epochs | Baseline training | 0.85 |
| 50 epochs | Extended training | 0.90 |
| 100 epochs | Further training | 0.90 |
| Close centers (1,1 vs 2,2) | Poor separation | 0.50 |
| Far centers (1,1 vs 4,5) | Clear separation | 0.90 |
| High-dim (100 D) | Better separability | 1.00 |

**Observations:**  
- Increasing the number of epochs improves accuracy up to a limit.  
- Greater inter-class distance leads to better separation.  
- Higher-dimensional data improves linear separability and accuracy.

---

## 🧠 Tech Stack

- **Language:** Python 3.10+  
- **Libraries:** NumPy, Matplotlib  
- **Dataset:** Synthetic 2D / 100D data generated via Gaussian noise  
- **Model Type:** Binary Perceptron with SGD updates  

---

## 🚀 Getting Started

1. **Clone Repository**
   ```bash
   git clone https://github.com/ali-rose/Bachelor_project.git
   cd Bachelor_project/Machine_learning/lab1
   ```

2. **Install Dependencies**
   ```bash
   pip install numpy matplotlib
   ```

3. **Run Experiment**
   ```bash
   python lab1.py
   ```

4. **View Results**
   - Generated decision-boundary plots: `1.png` – `6.png`  

---

## 🔗 Original Reference

This experiment is based on fundamental perceptron concepts from  
👉 *Rosenblatt (1958), The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain.*

---

## 📄 References

- NumPy Documentation — [https://numpy.org](https://numpy.org)  
- Matplotlib Documentation — [https://matplotlib.org](https://matplotlib.org)

---

## 🧩 Disclaimer

This project is for **educational and research purposes only**.  
All rights and data belong to their original authors.

---

## 👨‍💻 Author

**Ailixiaer Ailika**  
Bachelor Thesis Project — *Machine Learning Fundamentals (Lab 1)*  
📍 University Project Repository (Non-Commercial Use)

---

## 🪪 License

Released under the **MIT License**.  
Free to use, modify, and share with attribution.
