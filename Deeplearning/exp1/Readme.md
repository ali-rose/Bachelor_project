
## 📁 Project Directory

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

## 🧩 Project Overview

This project reproduces and extends **deep learning experiments** based on the *Fashion-MNIST* dataset from the book *Dive into Deep Learning (D2L)*.  
It includes:

- **Part 1:** Implementation of Softmax Regression from scratch.  
- **Part 2:** Multi-Layer Perceptron (MLP) experiments, testing the effects of hyperparameters, activation functions, and model depth.  
- **Additional Visuals:** PNG figures illustrating intermediate results and performance comparisons.  
- **Report:** A detailed written report summarizing experiment setup, analysis, and results.

---

## 🔬 Code Highlights

Key functionalities in [`exp1.py`](./exp1.py):

- **Dataset Loading:**  
  Automatically downloads and loads the Fashion-MNIST dataset for training and testing.  

- **Softmax Regression:**  
  Implements classification from scratch using matrix operations and cross-entropy loss.  

- **Evaluation Metrics:**  
  Custom accuracy calculation, training visualization (via `Animator` class), and prediction visualization.  

- **Multi-Layer Perceptron (MLP):**  
  Tests multiple architectures with varying:
  - Hidden layer sizes (`256`, `128-64-32`)
  - Number of hidden layers (2 vs 4)
  - Learning rates (`0.1`, `0.3`)
  - Activation functions (`ReLU`, `Sigmoid`)
  - Epochs (10–30)

- **Optimized Model:**  
  Final configuration achieves balanced training efficiency and accuracy using tuned learning rate and initialization.

---

## ⚙️ Tech Stack

- **Language:** Python 3.10+  
- **Frameworks:** PyTorch, D2L (Dive into Deep Learning)  
- **Visualization:** Matplotlib  
- **Tools:** Jupyter Notebook (`.ipynb`) for experiment tracking and analysis  

---

## 🚀 Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/ali-rose/Bachelor_project.git
   cd Bachelor_project/Deeplearning
   ```

2. **Install dependencies**

   ```bash
   pip install torch torchvision matplotlib d2l
   ```

3. **Run the experiment**

   ```bash
   python exp1.py
   ```

4. **View results**

   - Training logs will display in the console.
   - Output plots (e.g., loss & accuracy curves) will appear in the notebook or saved figures.
   - See `1.png`–`8.png` for visual results.

---


## 🧑‍💻 Author

**Ailixiaer Ailika**  
Bachelor Thesis Project — Deep Learning Fundamentals  
University Project Repository (Educational Use Only)

---

## 📄 License

This project is released under the **MIT License**.  
Feel free to use, modify, and share with proper attribution.
