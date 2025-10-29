# 🧠 Machine Learning Experiment — Naive Bayes Classification & KNN Noise Robustness (Lab 3)

📘 **Project Overview**  
This folder contains the implementation, visualization results, and detailed report for the **third machine learning experiment**, which focuses on **Naive Bayes classification** and **K-Nearest Neighbors (KNN)** under noisy data conditions.  
The goal is to evaluate **classification accuracy**, **error distribution**, and **noise robustness** between probabilistic and distance-based classifiers.

---

## ⚙️ Folder Structure

```plaintext
.
├── lab1.py                             # Training code for Requirement 1 (Naive Bayes)
├── lab2.py                             # Training code for the Extended Task (Noise + KNN)
├── 要求1-1.png                          # Requirement 1 – Training data distribution
├── 要求1-2.png                          # Requirement 1 – Naive Bayes prediction distribution
├── 要求1-3.png                          # Requirement 1 – Error rate visualization
├── 附加题1-1.png                        # Extra Task – Training data distribution (with noise)
├── 附加题1-2.png                        # Extra Task – Naive Bayes prediction distribution
├── 附加题1-3.png                        # Extra Task – KNN prediction distribution
├── 附加题1-4.png                        # Extra Task – Error rate comparison

```

---

## 🚀 Features

- 🧩 **Naive Bayes Classifier**
  - Implemented manually using Gaussian probability modeling.  
  - Calculates class priors, means, variances, and posterior probabilities.  
  - Assumes feature independence for simplicity and efficiency.

- ⚡ **K-Nearest Neighbors (KNN) Comparison**
  - Introduced noise points into the dataset to test robustness.  
  - Compared Naive Bayes and KNN under identical data conditions.  
  - Evaluated classification boundaries and noise sensitivity.

- 📊 **Evaluation Metrics**
  - Classification accuracy and overall error rate.  
  - Error visualization for both models (clean vs. noisy data).

---

## 🧩 Code Overview

### 🔹 Naive Bayes Implementation (lab1.py)

```python
class NaiveBayesClassifier:
    def __init__(self):
        self.priors, self.means, self.vars = {}, {}, {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            X_c = X[y == cls]
            self.priors[cls] = X_c.shape[0] / X.shape[0]
            self.means[cls] = X_c.mean(axis=0)
            self.vars[cls] = X_c.var(axis=0)

    def predict(self, X):
        preds = []
        for x in X:
            posteriors = []
            for cls in self.classes:
                prior = np.log(self.priors[cls])
                cond = -0.5 * np.sum(np.log(2 * np.pi * self.vars[cls]))
                cond -= 0.5 * np.sum(((x - self.means[cls]) ** 2) / self.vars[cls])
                posteriors.append(prior + cond)
            preds.append(self.classes[np.argmax(posteriors)])
        return np.array(preds)
```

### 🔹 Noise Injection & KNN Comparison (lab2.py)

```python
# Add random noise
noise_ratio = 0.1
num_noise = int(noise_ratio * n)
noise_X = np.random.rand(num_noise, 2) * 10
noise_Y = np.random.randint(1, 10, size=num_noise)

# Combine with original data
X_noisy = np.vstack([X, noise_X])
Y_noisy = np.concatenate([Y, noise_Y])

# KNN prediction
def knn_predict(X_train, y_train, x_test, k=5):
    distances = np.linalg.norm(X_train - x_test, axis=1)
    k_idx = np.argsort(distances)[:k]
    k_labels = y_train[k_idx]
    return np.bincount(k_labels).argmax()
```

---

## 📊 Experimental Findings

| Experiment | Method | Condition | Error Rate |
|-------------|---------|------------|-------------|
| Requirement 1 | Naive Bayes | Clean data | **1.35%** |
| Extra Task | Naive Bayes | Noisy data (10%) | **5.19%** |
| Extra Task | KNN (k=5) | Noisy data (10%) | **6.49%** |

**Observations:**  
- Both classifiers experience performance degradation when noise is added.  
- **Naive Bayes** demonstrates **stronger robustness** to noise due to probabilistic modeling.  
- **KNN** suffers more from mislabeled or overlapping noisy points since distance is directly affected.  

---

## 🧠 Tech Stack

- **Language:** Python 3.10+  
- **Libraries:** NumPy, Matplotlib  
- **Dataset:** Synthetic 2-D grid data with controlled noise ratios  
- **Models:** Gaussian Naive Bayes, K-Nearest Neighbors  
- **Environment:** Jupyter Notebook / Python CLI  

---

## 🚀 Getting Started

1. **Clone Repository**
   ```bash
   git clone https://github.com/ali-rose/Bachelor_project.git
   cd Bachelor_project/Machine_learning/lab3
   ```

2. **Install Dependencies**
   ```bash
   pip install numpy matplotlib
   ```

3. **Run the Experiments**
   ```bash
   python lab1.py   # Requirement 1 – Naive Bayes
   python lab2.py   # Extra Task – Noise + KNN Comparison
   ```

4. **View Results**
   - `要求1-*.png` → Clean data and Naive Bayes results  
   - `附加题1-*.png` → Noise robustness comparison (Naive Bayes vs KNN)  

---

## 🔗 Original Reference

This experiment follows fundamental Bayesian and KNN theory as introduced by:  
👉 *R. O. Duda, P. E. Hart & D. G. Stork (2001),* **Pattern Classification** (2nd ed.). Wiley.  

---

## 📄 References

- NumPy Documentation — [https://numpy.org](https://numpy.org)  
- Matplotlib Documentation — [https://matplotlib.org](https://matplotlib.org)  

---

## 🧩 Disclaimer

This project is for **educational and research use only**.  
All rights belong to their respective authors.

---

## 👨‍💻 Author

**Ailixiaer Ailika**  
Bachelor Thesis Project — *Machine Learning Fundamentals (Lab 3)*  
📍 University Project Repository (Non-Commercial Use)

---

## 🪪 License

Released under the **MIT License**.  
Free to use, modify, and redistribute with attribution.
