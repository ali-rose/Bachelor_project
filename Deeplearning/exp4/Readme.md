# 🧠 Deep Learning Experiment — Recurrent Neural Networks & Sequence Modeling (Lab 4)

📘 **Project Overview**  
This folder contains the code, Jupyter notebook, and output figures for the **fourth deep learning experiment**, focusing on **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory (LSTM)** models.  
The experiment builds and trains RNNs **from scratch** on the *Time Machine* dataset, comparing **random sampling vs. sequential sampling**, **basic RNN vs. LSTM**, and **custom gating mechanisms**.  
All models are implemented using **PyTorch** and **D2L (Dive into Deep Learning)**.

---

## ⚙️ Folder Structure

```plaintext
.
├── lab4.ipynb                                        # Interactive Jupyter notebook  
├── RNN不加随机抽样.png                                # RNN without random sampling  
├── RNN加随机抽样.png                                  # RNN with random sampling  
├── (附加题)RNN不加随机抽样.png                        # Extended task — RNN without random sampling  
├── （附加题）RNN加随机抽样.png                        # Extended task — RNN with random sampling  
├── LSTM.png                                          # LSTM experiment  
└── （附加题）LSTM.png                                 # Extended LSTM experiment  
```

---

## 🚀 Features

- **RNN and LSTM from Scratch**  
  - Built custom RNN and LSTM architectures without using PyTorch built-ins.  
  - Implemented hidden state initialization, time-step recurrence, and gradient clipping manually.  

- **Text Sequence Prediction**  
  - Trained models on the *Time Machine* dataset for character-level language modeling.  
  - Generated new text sequences based on given prefixes.  

- **Comparison Experiments**  
  - RNN with/without random sampling.  
  - Standard RNN vs. LSTM.  
  - Custom LSTM gate parameterization.  

- **Visualization**  
  - Training curves and perplexity evolution saved as `.png` images.  
  - Generated text outputs printed during training to evaluate prediction quality.

---

## 🧩 Code Overview

Core implementation (from [`lab4.ipynb`](./lab4.ipynb)):

```python
%matplotlib inline
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import collections, re

# -------------------- Data Loading --------------------
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# -------------------- RNN Model from Scratch --------------------
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(X @ W_xh + H @ W_hh + b_h)
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

class RNNModelScratch:
    """RNN implemented from scratch"""
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
```

---

### 🔁 Training and Prediction

```python
# Gradient Clipping
def grad_clipping(net, theta):
    """Clip gradients to prevent explosion"""
    params = [p for p in (net.params if not isinstance(net, nn.Module) else net.parameters()) if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

# Training Epoch
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Train for one epoch"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            for s in (state if isinstance(state, tuple) else [state]):
                s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

---

### 🧠 LSTM Implementation

```python
# LSTM parameter initialization
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape): return torch.randn(size=shape, device=device) * 0.01
    def three(): return (normal((num_inputs, num_hiddens)),
                         normal((num_hiddens, num_hiddens)),
                         torch.zeros(num_hiddens, device=device))
    # Input, forget, output, and candidate gates
    W_xi, W_hi, b_i = three()
    W_xf, W_hf, b_f = three()
    W_xo, W_ho, b_o = three()
    W_xc, W_hc, b_c = three()
    # Output layer
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o,
              W_xc, W_hc, b_c, W_hq, b_q]
    for p in params:
        p.requires_grad_(True)
    return params

# LSTM computation
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o,
     W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid(X @ W_xi + H @ W_hi + b_i)
        F = torch.sigmoid(X @ W_xf + H @ W_hf + b_f)
        O = torch.sigmoid(X @ W_xo + H @ W_ho + b_o)
        C_tilda = torch.tanh(X @ W_xc + H @ W_hc + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)
```

---

## ⚙️ Training Setup

```python
num_hiddens, num_epochs, lr = 256, 1000, 0.8
device = d2l.try_gpu()

# RNN Training (Sequential)
train_ch8(net, train_iter, vocab, lr, num_epochs, device)

# RNN Training (Random Sampling)
train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=True)

# LSTM Training
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device,
                            get_lstm_params, init_lstm_state, lstm)
train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

---

## 🧠 Tech Stack

- **Language:** Python 3.10+  
- **Frameworks:** PyTorch, D2L (Dive into Deep Learning)  
- **Dataset:** Time Machine (Character-level text corpus)  
- **Visualization:** Matplotlib  
- **Hardware:** GPU (via `d2l.try_gpu()`)

---

## 🚀 Getting Started

1. **Clone the Repository**

   ```bash
   git clone https://github.com/ali-rose/Bachelor_project.git
   cd Bachelor_project/Deeplearning/lab4
   ```

2. **Install Dependencies**

   ```bash
   pip install torch torchvision matplotlib d2l
   ```

3. **Run the Experiment**

   ```bash
   python lab4.py
   # or open lab4.ipynb for interactive execution
   ```

4. **View Results**

   - Training visualizations and perplexity plots are saved as `.png` files.  
   - Generated text predictions are printed during training.  
   - The `.docx` report summarizes findings and comparative analysis.

---

## 🔗 Original Reference

This experiment is based on the open-source textbook:  
👉 [Dive into Deep Learning (D2L.ai)](https://d2l.ai)

---

## 📄 References

- *PyTorch Documentation*: [https://pytorch.org](https://pytorch.org)  
- *D2L Official Repository*: [https://github.com/d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en)  
- *Hochreiter & Schmidhuber (1997)* — *Long Short-Term Memory*  

---

## 🧩 Disclaimer

This project is for **educational and research purposes only**.  
All rights belong to their respective authors.

---

## 👨‍💻 Author

**Ailixiaer Ailika**  
Bachelor Thesis Project — *Deep Learning Fundamentals (Lab 4)*  
📍 University Project Repository (Non-Commercial Use)

---

## 🪪 License

Released under the **MIT License**.  
Free to use, modify, and distribute with attribution.
