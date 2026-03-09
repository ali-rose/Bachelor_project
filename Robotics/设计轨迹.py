import numpy as np
import matplotlib.pyplot as plt

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

# 绘制曲线
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

for i in range(3):
    axs[0].plot(t, q[i], label=f'joint{i+1}')
    axs[1].plot(t, dq[i], label=f'joint{i+1}')
    axs[2].plot(t, ddq[i], label=f'joint{i+1}')

axs[0].set_title('joint_position')
axs[0].set_ylabel('position (rad or m)')
axs[0].grid(True)
axs[0].legend()

axs[1].set_title('joint_velocity')
axs[1].set_ylabel('velocity (rad/s or m/s)')
axs[1].grid(True)
axs[1].legend()

axs[2].set_title('joint_accelaration')
axs[2].set_ylabel('accelaration (rad/s^2 or m/s^2)')
axs[2].set_xlabel('time (s)')
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()
