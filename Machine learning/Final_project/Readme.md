```markdown
# 🎓 Machine Learning Final Assignment — Comprehensive Analysis

## 📘 Project Overview
This repository contains the source code, datasets, and final report for the **Machine Learning Final Assignment**. The project consists of three distinct experimental tasks covering **Time Series Prediction**, **Text Classification**, and **Risk Identification**.

The experiments compare various algorithms (XGBoost, ARIMA, SVM, Logistic Regression, Random Forest) to solve real-world data problems.

## ⚙️ Folder Structure

```text
.
├── pop-music/                          # Task 1: Music Trend Prediction
│   ├── lab1.py / lab2.py / lab3.py     # Source code for ARIMA & XGBoost
│   ├── mars_tianchi_songs.csv          # Song artist data
│   ├── mars_tianchi_user_actions.csv   # User playback history
│   └── artist_forecast_*.csv           # Prediction results
├── news/                               # Task 2: News Topic Classification
│   ├── lab1.py                         # NLP Classification script
│   ├── cnews.train.txt                 # Text dataset (50k records)
│   └── newslog.txt                     # Execution logs
├── risk trading/                       # Task 3: Risk Transaction Identification
│   ├── logistic.py                     # Logistic Regression model
│   ├── Xgboost.py                      # XGBoost model
│   ├── random_forest.py                # Random Forest model
│   ├── train.csv / pred.csv            # Training and prediction datasets
│   └── *_pred_results.csv              # Output predictions
├── 实验报告.pdf                         # Full Detailed Experiment Report (PDF)
└── requirements.txt                    # Project dependencies

```

## 🚀 Experiments & Features

### 1. 🎵 Music Trend Prediction (Time Series)

**Goal:** Predict the popularity trend of artists based on historical playback data from Alibaba Music.

* **Models Used:** XGBoost (Regression) vs. ARIMA.
* **Techniques:** Rolling window smoothing (7-day), Lag features, Time-series cross-validation.

| Model | Score (F-Value) | Outcome |
| --- | --- | --- |
| **XGBoost** | **8246.65** | ✅ **Best Performance** |
| ARIMA | -421.72 | Poor (Sensitive to zero-inflated data) |

### 2. 📰 News Topic Classification (NLP)

**Goal:** Classify news articles into 10 categories (Sports, Finance, Tech, etc.) using a dataset of 50,000 records.

* **Models Used:** Logistic Regression, SVM, Naive Bayes.
* **Techniques:** TF-IDF Vectorization, Label Encoding.

| Model | Test Accuracy | Observation |
| --- | --- | --- |
| **SVM** | **0.8264** | ✅ **Highest Accuracy** |
| Logistic Regression | 0.8202 | Competitive baseline |
| Naive Bayes | 0.7554 | Faster but less accurate |

### 3. 💳 Risk Transaction Identification (Binary Classification)

**Goal:** Identify risky transactions (Label 0/1) from daily trading details.

* **Models Used:** Logistic Regression, XGBoost, Random Forest.
* **Metric:** F1 Score (Harmonic mean of Precision and Recall).

| Model | F1 Score (Validation) | Performance |
| --- | --- | --- |
| **XGBoost** | **0.8644** | ✅ **Top Performer** |
| Random Forest | 0.8571 | Very close second |
| Logistic Regression | 0.7567 | Baseline performance |

## 🧠 Tech Stack

* **Language:** Python 3.x
* **Core Libraries:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (SVM, LR, RF, Naive Bayes)
* **Boosting:** `xgboost`
* **Time Series:** `statsmodels` (ARIMA)

## 🚀 Getting Started

1. **Clone Repository**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

```


2. **Install Dependencies**
```bash
pip install -r requirements.txt

```


3. **Run Experiments**
* **Music Prediction:** Navigate to `pop-music/` and run the scripts.
* **News Classification:** Navigate to `news/` and run `lab1.py`.
* **Risk Identification:** Navigate to `risk trading/` and run the specific model script (e.g., `Xgboost.py`).



## 👨‍💻 Author

**Ailixiaer Ailika **

## 🪪 License

This project is for educational purposes.

```

```
