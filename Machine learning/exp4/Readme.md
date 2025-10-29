# 🧠 Machine Learning Experiment — Support Vector Machine (SVM) Classification (Lab 4)

📘 **Project Overview**  
This folder contains the full implementation, visualization results, and report for the **fourth machine learning experiment**, which focuses on **Support Vector Machine (SVM)** classification using linear kernels.  
The study explores how **data separability**, **margin width**, and **regularization parameter C** influence the decision boundary, support vectors, and overall classification performance.

---

## ⚙️ Folder Structure

```plaintext
.
├── lab1.py                             # Requirement 1: Basic linear SVM training (C = 1e5)
├── lab2.py                             # Extended task: parameter tuning for C (0.01 – 10000)
├── log.txt                             # Support-vector b-values for each C in the extended task
├── 要求1-1.png                          # Center (6, 6): training data distribution
├── 要求1-2.png                          # Center (6, 6): prediction distribution
├── 要求1-3.png                          # Center (6, 6): support-vector b-values
├── 要求1-4.png                          # Center (3, 3): training data distribution
├── 要求1-5.png                          # Center (3, 3): prediction distribution
├── 要求1-6.png                          # Center (3, 3): support-vector b-values
├── 附加题3-1.png                        # Center (6, 6): predictions under C = 0.01, 0.1, 1, 10000
├── 附加题3-2.png                        # Center (3, 3): predictions under C = 0.01, 0.1, 1, 10000
```

---

## 🚀 Features

- ⚙️ **Linear SVM Implementation**
  - Uses `sklearn.svm.SVC(kernel='linear')` for clear visualization of hyperplanes.  
  - Displays classification surfaces, margin boundaries, and all support vectors.

- 🧩 **Two Experimental Scenarios**
  1. **Requirement 1:**  
     - Second-class center = (6, 6) and (3, 3).  
     - Visualizes data distribution, classification boundary, and computed b-values.
  2. **Extended Task 3:**  
     - Investigates the effect of different C values (0.01 / 0.1 / 1 / 10000).  
     - Shows how margin width and misclassification tolerance change.  
     - Logs all support-vector b-values to `log.txt`.

- 📈 **Analysis Highlights**
  - Smaller C → larger margin, higher tolerance → potential underfitting.  
  - Larger C → narrower margin, stricter constraint → risk of overfitting.  
  - Support vectors remain critical for defining the decision boundary.

---

## 🧩 Code Overview

### 🔹 Requirement 1 (lab1.py)

```python
from sklearn.svm import SVC
import numpy as np, matplotlib.pyplot as plt

n = 100
center1, center2 = np.array([1,1]), np.array([6,6])
X = np.vstack([center1 + np.random.randn(n,2),
               center2 + np.random.randn(n,2)])
Y = np.hstack([np.ones(n), -np.ones(n)])

clf = SVC(kernel='linear', C=1e5)
clf.fit(X, Y)

w, b = clf.coef_[0], clf.intercept_[0]
support_vectors = clf.support_vectors_
calculated_bs = Y[clf.support_] - np.dot(support_vectors, w)
print("支持向量计算出的 b 值:", calculated_bs)
```

### 🔹 Extended Task 3 (lab2.py)

```python
def plot_svm_with_different_C(C_values, X, Y):
    for i, C in enumerate(C_values):
        clf = SVC(kernel='linear', C=C)
        clf.fit(X, Y)
        w, b = clf.coef_[0], clf.intercept_[0]
        # Plot decision boundary & margins …
C_values = [0.01, 0.1, 1, 10000]
plot_svm_with_different_C(C_values, X, Y)
```

---

## 📊 Experimental Findings

| Scenario | Data Center | C Value Range | Key Observation | Margin Behavior |
|-----------|--------------|---------------|-----------------|----------------|
| Requirement 1 | (6, 6) / (3, 3) | 1e5 | Stable separation; support vectors clearly defined | Wide margin |
| Extended Task 3 | (6, 6) / (3, 3) | 0.01 – 10000 | C ↑ → margin ↓, error ↓, risk of overfitting | Adjustable |

**Summary of B-values (log.txt):**  
- Smaller C → B values more scattered (soft margin tolerance).  
- Larger C → B values converge (closer to hard margin solution).  

---

## 🧠 Tech Stack

- **Language:** Python 3.10+  
- **Libraries:** NumPy, Matplotlib, scikit-learn  
- **Algorithm:** Support Vector Machine (SVM, Linear Kernel)  
- **Environment:** Jupyter Notebook / Python CLI  

---

## 🚀 Getting Started

1. **Clone Repository**
   ```bash
   git clone https://github.com/ali-rose/Bachelor_project.git
   cd Bachelor_project/Machine_learning/lab4
   ```

2. **Install Dependencies**
   ```bash
   pip install numpy matplotlib scikit-learn
   ```

3. **Run Experiments**
   ```bash
   python lab1.py   # Requirement 1 – basic SVM
   python lab2.py   # Extended Task 3 – C-parameter tuning
   ```

4. **View Results**
   - Images `要求1-*.png` → center (6, 6) and (3, 3) comparisons  
   - Images `附加题3-*.png` → C-parameter visualization  
   - `log.txt` → support-vector b-values  

---

## 🔗 Original Reference

Based on the foundational work of:  
👉 *Cortes, C. & Vapnik, V. (1995). “Support Vector Networks.” Machine Learning 20(3): 273-297.*

---

## 📄 References

- scikit-learn Documentation — [https://scikit-learn.org](https://scikit-learn.org)  
- NumPy Documentation — [https://numpy.org](https://numpy.org)  
- Matplotlib Documentation — [https://matplotlib.org](https://matplotlib.org)  

---

## 🧩 Disclaimer

This project is for **educational and research purposes only**.  
All rights belong to their respective authors.

---

## 👨‍💻 Author

**Ailixiaer Ailika**  
Bachelor Thesis Project — *Machine Learning Fundamentals (Lab 4)*  
📍 University Project Repository (Non-Commercial Use)

---

## 🪪 License

Released under the **MIT License**.  
Free to use, modify, and redistribute with attribution.
