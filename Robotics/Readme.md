```markdown
# 🤖 Robotics Final Project — SCARA Robot Control & Simulation

## 📘 Project Overview
This repository contains the source code, simulation results, and the final report for the **Robotics** course project. The experiment focuses on the modeling, trajectory planning, and control system design for a simplified **3-DOF SCARA Robot**.

The project implements and compares three different control strategies: **PD Control**, **Inverse Dynamics Control**, and **Impedance Control**, analyzing their performance under standard conditions and external force disturbances.

## ⚙️ File Structure

The project files are organized as follows:

```text
.
├── 报告.pdf                        # 📄 Final Detailed Experiment Report
├── 设计轨迹.py                     # 🐍 Script: Cubic polynomial trajectory planning (0-10s)
├── 末端位姿.py                     # 🐍 Script: End-effector pose calculation & visualization
├── 末端速度曲线.py                 # 🐍 Script: End-effector velocity profile analysis
├── PD控制器.py                     # 🐍 Script: PD Controller simulation
├── 逆动力学控制器.py               # 🐍 Script: Inverse Dynamics Controller simulation
├── 阻抗控制器.py                   # 🐍 Script: Impedance Controller simulation
├── 施加外力的控制器.py             # 🐍 Script: Impedance Control with external force disturbance
├── *.png                           # 📊 Generated result plots (Visualizations)
└── README.md                       # 📝 Project Documentation

```

## 🚀 Features & Experiments

### 1. 🦾 Robot Modeling

**Model:** Simplified SCARA Robot (2 Rotational + 1 Prismatic joints).

* **Kinematics:** DH Parameter derivation, Forward/Inverse Kinematics.
* **Dynamics:** Jacobian matrix calculation and Lagrangian dynamics modeling ($\tau = M(q)\ddot{q} + V(q,\dot{q}) + G(q)$).

### 2. 📈 Trajectory Planning

**Goal:** Plan a smooth path using cubic polynomials.

* **Phase 1 (0-5s):** Start ($0,0,0$) $\to$ Mid ($20^\circ, 40^\circ, 0.1m$).
* **Phase 2 (5-10s):** Mid $\to$ End ($30^\circ, 60^\circ, 0.15m$).
* **Outputs:** Joint positions, velocities, and accelerations.

### 3. 🎛️ Control Strategies

The project implements three controllers to track the planned trajectory:

| Controller | Description | Key Findings |
| --- | --- | --- |
| **PD Controller** | Feedback control based on error ($u = K_p e + K_d \dot{e}$). | Simple to implement but sensitive to model uncertainty and disturbances. |
| **Inverse Dynamics** | Uses the dynamic model to cancel non-linearities. | **High accuracy** tracking; excellent resistance to external disturbances. |
| **Impedance Control** | Controls the dynamic relationship between force and position. | **Compliant behavior**; maintains stability and adapts well when external forces are applied. |

### 4. ⚡ Disturbance Testing

**Scenario:** Applied external force vector $F_{ext} = [2, 2, 3] N$ during $t=4s$ to $t=6s$.

* **Observation:** The Impedance Controller successfully absorbed the impact, demonstrating compliance, whereas rigid controllers might oscillate or fail.

## 🧠 Tech Stack

* **Language:** Python 3.x
* **Libraries:**
* `numpy` (Matrix operations, Linear Algebra)
* `matplotlib` (2D/3D Plotting, Visualization)



## 🚀 Getting Started

1. **Clone Repository**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

```


2. **Install Dependencies**
```bash
pip install numpy matplotlib

```


3. **Run Simulations**
* To view trajectory planning:
```bash
python 设计轨迹.py

```


* To run the Inverse Dynamics simulation:
```bash
python 逆动力学控制器.py

```


* To test the Impedance Controller with external force:
```bash
python 施加外力的控制器.py

```





## 👨‍💻 Author

**Ailixiaer Ailika **
*College of Artificial Intelligence, Nankai University*

## 🪪 License

This project is for educational and research purposes.

```

```
