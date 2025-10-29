# 🧠 Machine Learning Experiment — K-Means Clustering and Distance Metric Analysis (Lab 5)

📘 **Project Overview**  
This folder contains the full implementation, visualization results, and written report for the **fifth machine learning experiment**, which focuses on **K-Means clustering** using different **distance metrics**.  
The experiment explores how **Euclidean** and **Manhattan** distances affect clustering performance and how **data distribution** and **initialization** influence final results.  
Additionally, an **extended task** investigates clustering performance on a more complex dataset.

---

## ⚙️ Folder Structure

```plaintext
.
├── lab1.py                             # Requirement 1 – K-Means clustering (Euclidean & Manhattan)
├── lab2.py                             # Extended Task – Complex dataset clustering
├── 要求1-1.png                          # Unclustered data
├── 要求1-2.png – 要求1-6.png            # Euclidean distance clustering results
├── 要求1-7.png – 要求1-11.png           # Manhattan distance clustering results
├── 附加题1-1.png                        # Extended dataset – unclustered data
├── 附加题1-2.png – 附加题1-5.png        # Euclidean distance clustering results (complex data)
├── 附加题1-6.png – 附加题1-8.png        # Manhattan distance clustering results (complex data)
```

---

## 🚀 Features

- ⚙️ **K-Means Implementation (Custom & scikit-learn)**  
  - Clustering on synthetic 2D datasets with adjustable cluster counts (K).  
  - Supports both **Euclidean** and **Manhattan** distance metrics.  
  - Visualizes unclustered data, cluster assignments, and centroid positions.

- 🧩 **Two Experiment Stages**
  1. **Requirement 1:**  
     - 9-cluster dataset using grid-like data distribution.  
     - Distance metrics:  
       - *Euclidean* → Figures `要求1-2`–`要求1-6`  
       - *Manhattan* → Figures `要求1-7`–`要求1-11`  
     - Shows clustering convergence and centroid movements.
  2. **Extended Task (lab2.py):**  
     - Complex dataset with irregular rectangular regions.  
     - Distance metrics:  
       - *Euclidean* → Figures `附加题1-2`–`附加题1-5`  
       - *Manhattan* → Figures `附加题1-6`–`附加题1-8`  
     - Demonstrates performance degradation in non-spherical data.

---

## 🧩 Code Overview

### 🔹 Requirement 1 (lab1.py)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min

def kmeans_custom(X, K, metric='euclidean', max_iter=100):
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    for _ in range(max_iter):
        labels, _ = pairwise_distances_argmin_min(X, centroids, metric=metric)
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels, centroids
```

### 🔹 Extended Task (lab2.py)
```python
metrics = ['euclidean', 'manhattan']
for metric in metrics:
    labels, centroids = kmeans_custom(X, K=6, metric=metric)
    plt.scatter(X[:,0], X[:,1], c=labels, s=10)
    plt.scatter(centroids[:,0], centroids[:,1], c='m', marker='s', s=100)
    plt.title(f'K-Means Clustering with {metric.title()} Distance')
    plt.show()
```

---

## 📊 Experimental Findings

| Metric | Data Type | Performance | Convergence | Visual Shape |
|---------|------------|--------------|--------------|---------------|
| Euclidean | Regular grid | Fast | Stable | Circular clusters |
| Manhattan | Regular grid | Moderate | Stable | Diamond clusters |
| Euclidean | Complex dataset | Unstable | Sensitive to init | Non-spherical errors |
| Manhattan | Complex dataset | Better robustness | Slight distortion | Rectangular boundaries |

**Key Insights**
- Euclidean distance assumes spherical, isotropic clusters.  
- Manhattan distance performs better with rectangular or sparse data.  
- K-Means is sensitive to initial centroid positions.  
- The algorithm struggles on irregularly shaped or uneven-density data.

---

## 🧠 Tech Stack

- **Language:** Python 3.10+  
- **Libraries:** NumPy, Matplotlib, scikit-learn  
- **Algorithm:** K-Means (custom + sklearn version)  
- **Metrics:** Euclidean Distance, Manhattan Distance  
- **Environment:** Jupyter Notebook / Python CLI  

---

## 🚀 Getting Started

1. **Clone Repository**
   ```bash
   git clone https://github.com/ali-rose/Bachelor_project.git
   cd Bachelor_project/Machine_learning/lab5
   ```

2. **Install Dependencies**
   ```bash
   pip install numpy matplotlib scikit-learn
   ```

3. **Run Experiments**
   ```bash
   python lab1.py   # Requirement 1 – distance metric comparison
   python lab2.py   # Extended Task – complex data clustering
   ```

4. **View Results**
   - `要求1-*.png` → Euclidean & Manhattan clustering (regular dataset)  
   - `附加题1-*.png` → Clustering on complex dataset  

---

## 🔗 Original Reference

Based on the classical K-Means algorithm:  
👉 *MacQueen, J. (1967). “Some Methods for Classification and Analysis of Multivariate Observations.” Proc. 5th Berkeley Symposium on Math. Statistics and Probability.*

---

## 📄 References

- scikit-learn Documentation — [https://scikit-learn.org](https://scikit-learn.org)  
- NumPy Documentation — [https://numpy.org](https://numpy.org)  
- Matplotlib Documentation — [https://matplotlib.org](https://matplotlib.org)  

---

## 🧩 Disclaimer

This project is for **educational and research use only**.  
All rights belong to their original authors.

---

## 👨‍💻 Author

**Ailixiaer Ailika**  
Bachelor Thesis Project — *Machine Learning Fundamentals (Lab 5)*  
📍 University Project Repository (Non-Commercial Use)

---

## 🪪 License

Released under the **MIT License**.  
Free to use, modify, and redistribute with attribution.
