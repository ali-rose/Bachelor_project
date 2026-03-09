import numpy as np
import matplotlib.pyplot as plt

# 设定PD控制器增益
Kp = np.array([100, 100, 100])
Kd = np.array([20, 20, 20])

# 时间节点
t0 = 0
tmid = 5
tf = 10


# 位置节点
q0 = np.array([0, 0, 0])
qmid = np.array([np.radians(20), np.radians(40), 0.1])
qf = np.array([np.radians(30), np.radians(60), 0.15])

# 初始化结果
t1 = np.linspace(t0, tmid, 100)
t2 = np.linspace(tmid, tf, 100)
q = []
dq = []
ddq = []

# 分别计算每个关节的轨迹
for i in range(3):
    # 第一段轨迹规划系数
    A1 = np.array([
        [1, t0, t0**2, t0**3],
        [0, 1, 2*t0, 3*t0**2],
        [1, tmid, tmid**2, tmid**3],
        [0, 1, 2*tmid, 3*tmid**2]
    ])
    B1 = np.array([q0[i], 0, qmid[i], 0])
    coeffs1 = np.linalg.solve(A1, B1)

    # 第二段轨迹规划系数
    A2 = np.array([
        [1, tmid, tmid**2, tmid**3],
        [0, 1, 2*tmid, 3*tmid**2],
        [1, tf, tf**2, tf**3],
        [0, 1, 2*tf, 3*tf**2]
    ])
    B2 = np.array([qmid[i], 0, qf[i], 0])
    coeffs2 = np.linalg.solve(A2, B2)

    # 位置曲线
    q1 = np.polyval(coeffs1[::-1], t1)
    q2 = np.polyval(coeffs2[::-1], t2)

    # 速度曲线
    dq1 = np.polyval(np.polyder(coeffs1[::-1]), t1)
    dq2 = np.polyval(np.polyder(coeffs2[::-1]), t2)

    # 加速度曲线
    ddq1 = np.polyval(np.polyder(coeffs1[::-1], 2), t1)
    ddq2 = np.polyval(np.polyder(coeffs2[::-1], 2), t2)

    # 合并数据
    q.append(np.concatenate((q1, q2)))
    dq.append(np.concatenate((dq1, dq2)))
    ddq.append(np.concatenate((ddq1, ddq2)))

# 转换为numpy数组
q = np.array(q)
dq = np.array(dq)
ddq = np.array(ddq)

# 时间范围
t = np.concatenate((t1, t2))

dt = t[1] - t[0]

# 期望轨迹
q_des = q
dq_des = np.gradient(q, axis=1) / dt
ddq_des = np.gradient(dq_des, axis=1) / dt

# 初始化
q_actual = np.zeros_like(q)
dq_actual = np.zeros_like(dq)
ddq_actual = np.zeros_like(ddq)
u = np.zeros_like(q)

# 初始条件
q_actual[:, 0] = q[:, 0]
dq_actual[:, 0] = dq[:, 0]

# 仿真
for i in range(1, len(t)):
    # 计算控制输入
    u[:, i - 1] = Kp * (q_des[:, i - 1] - q_actual[:, i - 1]) + Kd * (dq_des[:, i - 1] - dq_actual[:, i - 1])

    # 计算加速度
    ddq_actual[:, i - 1] = u[:, i - 1]  # 简化为直接等于控制输入

    # 计算速度和位置
    dq_actual[:, i] = dq_actual[:, i - 1] + ddq_actual[:, i - 1] * dt
    q_actual[:, i] = q_actual[:, i - 1] + dq_actual[:, i - 1] * dt

# 最后一个时刻的控制量和加速度无法计算，赋值为最后一个已知值
u[:, -1] = u[:, -2]
ddq_actual[:, -1] = ddq_actual[:, -2]

# 绘制结果曲线
fig, axs = plt.subplots(4, 1, figsize=(10, 12))

# 关节位置
for i in range(3):
    axs[0].plot(t, q_des[i], 'r--', label=f'Desired Joint {i + 1}' if i == 0 else "")
    axs[0].plot(t, q_actual[i], label=f'Actual Joint {i + 1}')
axs[0].set_ylabel('position (rad or m)')
axs[0].set_title('joint_position')
axs[0].grid(True)
axs[0].legend()

# 关节速度
for i in range(3):
    axs[1].plot(t, dq_des[i], 'r--', label=f'Desired Joint {i + 1}' if i == 0 else "")
    axs[1].plot(t, dq_actual[i], label=f'Actual Joint {i + 1}')
axs[1].set_ylabel('velocity (rad/s or m/s)')
axs[1].set_title('joint_velocity')
axs[1].grid(True)
axs[1].legend()

# 关节加速度
for i in range(3):
    axs[2].plot(t, ddq_des[i], 'r--', label=f'Desired Joint {i + 1}' if i == 0 else "")
    axs[2].plot(t, ddq_actual[i], label=f'Actual Joint {i + 1}')
axs[2].set_ylabel('accelaration (rad/s^2 or m/s^2)')
axs[2].set_title('joint_accelaration')
axs[2].grid(True)
axs[2].legend()

# 控制量
for i in range(3):
    axs[3].plot(t, u[i], label=f'Control Input {i + 1}')
axs[3].set_ylabel('controlled quantity (N or Nm)')
axs[3].set_title('controlled quantity')
axs[3].set_xlabel('time (s)')
axs[3].grid(True)
axs[3].legend()

plt.tight_layout()
plt.show()
