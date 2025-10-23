# 🌿 XGBoost-Based Air Pollutant Prediction System

This repository contains the implementation of a **machine learning–based air quality prediction system**, integrating XGBoost regression models for pollutant concentration forecasting and a **PyQt5-based graphical user interface (GUI)** for real-time prediction.

---

## 📘 Project Overview

This project aims to build and deploy an **XGBoost regression model** to predict **ozone (O₃)** and **nitrogen oxides (NOₓ)** concentrations based on meteorological and emission features.
It provides both a **model training pipeline** and an **interactive prediction tool** with user-friendly input and visualization.

**Key Functions:**

* Train XGBoost models using environmental datasets;
* Evaluate model accuracy with MAE, RMSE, and MAPE metrics;
* Save and load pre-trained `.bin` models for fast deployment;
* Provide an easy-to-use GUI for real-time air quality prediction.

---

## ⚙️ Folder Structure

```
├── mainmulti.py                # Model training and evaluation script
├── ui.py                       # PyQt5 GUI for real-time prediction
└── requirements.txt             # Project dependencies
```

---

## 🚀 Features

* **Model Training and Evaluation**

  * Uses `XGBRegressor` with optimized hyperparameters for regression tasks.
  * Evaluates performance using `MAE`, `RMSE`, and `MAPE`.
  * Visualizes feature importance and prediction accuracy.

* **Interactive GUI Prediction**

  * Built with **PyQt5**, allowing users to select models and input temperature, humidity, wind, and emission values.
  * Displays real-time prediction results.
  * Supports model switching (Ozone vs. NOx).

* **Model Persistence**

  * Models are saved and loaded in `.bin` format via `XGBoost.save_model()` and `load_model()`.

---

## 💻 Usage

### 1️⃣ Train the Model

To train the ozone prediction model and visualize its results:

```bash
python mainmulti.py
```

This script will:

* Load the dataset `finalmulti(Ozone).xlsx`;
* Split the data into training and testing sets;
* Train an XGBoost model;
* Output performance metrics;
* Save the model as `xgboost_model_ozone.bin`.

---

### 2️⃣ Run the GUI for Prediction

To start the prediction interface:

```bash
python ui.py
```

You can:

* Choose between **Ozone** and **NOx** models;
* Input temperature, humidity, wind speed, and emission data;
* Click “预测” to display predicted pollutant concentration;
* Click “清零” to reset all input fields.

---

## 🧠 Model Information

* **Algorithm:** XGBoost (Gradient Boosted Decision Trees)
* **Input Features:** Temperature, Humidity, Wind, Emission
* **Output Variables:** Ozone or Nitrogen Oxides concentration
* **Evaluation Metrics:** MAE, MAPE, RMSE
* **Visualization:** Matplotlib-based feature importance and result plots

---

## 🧬 Dependencies

Install all dependencies with:

```bash
pip install -r requirements.txt
```

**Main Libraries:**

* numpy
* pandas
* scikit-learn
* xgboost
* matplotlib
* PyQt5
* openpyxl
* joblib

---

## 📊 Example Output

**Model Evaluation Metrics (example):**

```
MAE  :  1.82
MAPE :  0.072
RMSE :  2.46
```

**Feature Importance Visualization:**
Displays a horizontal bar chart of feature importances from the trained model.

---

## 📝 References

* [XGBoost Official Documentation](https://xgboost.readthedocs.io/)
* [PyQt5 Official Guide](https://doc.qt.io/qtforpython/)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## ⚠️ Notes

* This project is for **research and educational purposes only**.
* The dataset and trained models are used to illustrate the integration of machine learning with GUI-based prediction systems.
* All copyrights belong to their respective authors.
