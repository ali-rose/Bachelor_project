# 💡 LightGBM-Based Air Pollutant Prediction System

This repository implements a **LightGBM regression-based air quality prediction system**, integrating pollutant concentration modeling with a **PyQt5 graphical user interface (GUI)** for real-time prediction and analysis.

---

## 📘 Project Overview

This project leverages **LightGBM** (Light Gradient Boosting Machine) to predict **ozone (O₃)** and **nitrogen oxides (NOₓ)** levels using environmental data such as temperature, humidity, wind speed, and emission rates.

It provides both model training and deployment functionalities, combining powerful gradient boosting with a user-friendly prediction interface.

**Core Functions:**

* Train and evaluate LightGBM regression models for multiple pollutants.
* Export trained models to `.bin` format for reuse.
* Provide an interactive GUI for quick pollutant concentration prediction.

---

## ⚙️ Folder Structure

```
├── main.py                   # LightGBM model training and evaluation script
├── ui.py                     # PyQt5 GUI for real-time prediction
```

---

## 🚀 Features

* **Efficient Model Training**

  * Uses LightGBM's fast histogram-based gradient boosting algorithm.
  * Supports multi-feature training (Temperature, Humidity, Wind, Emission).
  * Saves models in `.bin` format for lightweight deployment.

* **Model Evaluation**

  * Calculates standard regression metrics: MAE, MAPE, and RMSE.
  * Visualizes performance trends and feature importance using Matplotlib.

* **Interactive GUI**

  * Built with **PyQt5** for intuitive operation.
  * Users can select pollutant models (Ozone / NOx) and input key environmental factors.
  * Provides instant pollutant concentration predictions.

---

## 💻 Usage

### 1️⃣ Train the Model

Run the main training script to train the LightGBM regression model:

```bash
python main.py
```

This will:

* Load the dataset.
* Train the LightGBM model.
* Evaluate its performance using MAE, MAPE, and RMSE.
* Save trained models as `.bin` files.

---

### 2️⃣ Run the GUI for Prediction

Launch the GUI for real-time prediction:

```bash
python ui.py
```

**Usage:**

* Choose between **Ozone** and **NOx** models.
* Enter the environmental parameters.
* Click **“预测” (Predict)** to get predicted pollutant levels.
* Click **“清零” (Clear)** to reset all fields.

---

## 🧠 Model Information

* **Algorithm:** LightGBM (Gradient Boosted Decision Trees)
* **Input Features:** Temperature, Humidity, Wind, Emission
* **Output Variables:** Ozone or Nitrogen Oxides concentration
* **Evaluation Metrics:** MAE, MAPE, RMSE
* **Visualization:** Feature importance and prediction trend plots

---

**Main Libraries:**

* numpy
* pandas
* lightgbm
* scikit-learn
* matplotlib
* PyQt5
* openpyxl

---

## 📊 Example Output

**Sample Evaluation Metrics:**

```
MAE  :  1.65
MAPE :  0.068
RMSE :  2.34
```

**Feature Importance Plot:**
Displays ranked features influencing the pollutant concentration prediction.

---

## 📄 References

* [LightGBM Official Documentation](https://lightgbm.readthedocs.io/)
* [PyQt5 Official Guide](https://doc.qt.io/qtforpython/)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## ⚠️ Notes

* This project is for **research and educational purposes only**.
* The datasets and models are designed to demonstrate machine learning-based environmental prediction.
* All copyrights belong to their original authors and contributors.
