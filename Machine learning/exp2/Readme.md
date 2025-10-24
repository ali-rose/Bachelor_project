# 🧠 Machine Learning Experiment — K-Nearest Neighbors (KNN) Classification (Lab 2)

📘 **Project Overview**  
This folder contains the complete implementation, visualization results, and written report for the **second machine learning experiment**, which studies the **K-Nearest Neighbors (KNN)** algorithm.  
The experiment investigates how **distance metrics**, **training sample size**, and **the number of neighbors K** influence classification accuracy and generalization on synthetic two-dimensional data.

---

## ⚙️ Folder Structure

```plaintext
.
├── lab1.py                               # Code for Requirement 1, Extra 1, Extra 2
├── lab2.py                               # Code for Extra 3 (K-value analysis)
├── 要求1-1.png                            # Requirement 1 – Training data distribution
├── 要求1-2.png                            # Requirement 1 – Test data distribution
├── 要求1-3.png                            # Requirement 1 – Predicted data distribution
├── 要求1-4.png                            # Requirement 1 – Error-rate visualization
├── 附加题1-1.png                          # Extra 1 – Training data distribution
├── 附加题1-2.png                          # Extra 1 – Test data distribution
├── 附加题1-3.png                          # Extra 1 – Predicted data distribution
├── 附加题1-4.png                          # Extra 1 – Error-rate visualization
├── 附加题2-1.png – 附加题2-4.png            # Extra 2 – Manhattan distance: train/test/predict/error
├── 附加题2-5.png – 附加题2-8.png            # Extra 2 – Chebyshev distance: train/test/predict/error
├── 附加题3-1.png                          # Extra 3 – Training data distribution
├── 附加题3-2.png                          # Extra 3 – Error rate vs K curve
```

---

## 🚀 Features

- 🧩 **KNN Implementation (from scratch)**  
  - Manual computation of Euclidean, Manhattan, and Chebyshev distances.  
  - Majority-vote prediction without external ML libraries.

- 🔍 **Experiments Conducted**
  1. **Requirement 1:**  
     Baseline KNN classifier using Euclidean distance.  
     Images 1–4 → training / testing / prediction / error rate.
  2. **Extra 1:**  
     Reduced training sample size to analyze generalization behavior (100 train vs 500 test).  
     Images 1–4 → training / testing / prediction / error rate.
  3. **Extra 2:**  
     Comparison of **Manhattan** vs **Chebyshev** distances.  
     - 1–4 → Manhattan results  
     - 5–8 → Chebyshev results.
  4. **Extra 3:**  
     Influence of **K value** (1–100) on accuracy.  
     Images 1–2 → training data and K-error curve.

- 📈 **Evaluation Metrics**  
  - Classification accuracy and error rate.  
  - Visual comparison of decision boundaries.

---

## 🧩 Code Overview

### 🔹 Core KNN Function
```python
def k_nearest_neighbors(X_train, y_train, x_test, k, dist_fn):
    distances = [dist_fn(x_train, x_test) for x_train in X_train]
    k_idx = np.argsort(distances)[:k]
    labels = y_train[k_idx]
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]
```

### 🔹 Distance Metrics
```python
def euclidean(a, b):  return np.sqrt(np.sum((a - b)**2))
def manhattan(a, b):  return np.sum(np.abs(a - b))
def chebyshev(a, b):  return np.max(np.abs(a - b))
```

### 🔹 Error-Rate Calculation
```python
pred = [k_nearest_neighbors(X_train, y_train, x, k, euclidean) for x in X_test]
error = np.mean(pred != y_test)
print(f"Error rate: {error:.3f}")
```

---

## 📊 Experimental Findings

| Experiment | Distance | Training / Test | Best K | Error Rate |
|-------------|-----------|-----------------|--------|-------------|
| Requirement 1 | Euclidean | 500 / 100 | 3 | 0.00 |
| Extra 1 | Euclidean | 100 / 500 | 3 | 0.05 |
| Extra 2-1 | Manhattan | 100 / 500 | 5 | 0.03 |
| Extra 2-2 | Chebyshev | 100 / 500 | 5 | 0.11 |
| Extra 3 | Euclidean | 500 / 100 | 1–100 | Best ≈ K = 3–5 |

**Key Insights**
- Increasing K smooths boundaries but may underfit.  
- Manhattan distance performs better on sparse grids.  
- Too few training samples increase variance and error.  
- Chebyshev distance exaggerates axis-aligned bias.

---

## 🧠 Tech Stack

- **Language:** Python 3.10+  
- **Libraries:** NumPy, Matplotlib  
- **Dataset:** Synthetic 2-D data (0–10 grid)  
- **Algorithm:** K-Nearest Neighbors (Classic Model)  
- **Environment:** Jupyter Notebook / Command Line  

---

## 🚀 Getting Started

1. **Clone Repository**
   ```bash
   git clone https://github.com/ali-rose/Bachelor_project.git
   cd Bachelor_project/Machine_learning/lab2
   ```
2. **Install Dependencies**
   ```bash
   pip install numpy matplotlib
   ```
3. **Run Experiments**
   ```bash
   python lab1.py   # Requirement 1 – Extra 2
   python lab2.py   # Extra 3 (K-value analysis)
   ```
4. **View Results**
   - Visuals → `要求1-*.png`, `附加题*-*.png`  

---

## 🔗 Original Reference

Based on the classic KNN framework from  
👉 *T. M. Cover & P. E. Hart (1967), “Nearest Neighbor Pattern Classification,” IEEE Trans. Info. Theory.*

---

## 📄 References

- NumPy Docs — [https://numpy.org](https://numpy.org)  
- Matplotlib Docs — [https://matplotlib.org](https://matplotlib.org)

---

## 🧩 Disclaimer

This project is for **educational and research use only**.  
All rights belong to their original authors.

---

## 👨‍💻 Author

**Ailixiaer Ailika**  
Bachelor Thesis Project — *Machine Learning Fundamentals (Lab 2)*  
📍 University Project Repository (Non-Commercial Use)

---

## 🪪 License

Released under the **MIT License**.  
Free to use, modify, and redistribute with attribution.
