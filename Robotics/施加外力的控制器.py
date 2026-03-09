import numpy as np
import matplotlib.pyplot as plt

# 初始位置和期望位置
q0 = np.array([0, 0, 0])
qd = np.array([np.radians(30), np.radians(60), 0.15])

# 期望末端刚性和阻尼参数
K_X = np.diag([500, 500, 500])
K_B = np.diag([50, 50, 30])

# 动力学参数
m1, m2, m3 = 2.431, 3.7860, 0.5552 # 假设各连杆质量为1kg
I1, I2 = 0.03219, 0.03376  # 假设各连杆惯性为1kg*m^2
g = 9.81  # 重力加速度
l1,l2,l3=0.25, 0.25, 0.245

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


# 定义雅可比矩阵的计算函数
def jacobian(q):
    theta1, theta2, d3 = q
    J = np.array([
        [-l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2), -l2 * np.sin(theta1 + theta2), 0],
        [l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2), l2 * np.cos(theta1 + theta2), 0],
        [0, 0, -1]
    ])
    return J


# 计算末端位置的函数
def forward_kinematics(q):
    theta1, theta2, d3 = q
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    z = d3
    return np.array([x, y, z])


# 初始条件
q_actual = np.zeros((3, len(t)))
dq_actual = np.zeros((3, len(t)))
ddq_actual = np.zeros((3, len(t)))
tau = np.zeros((3, len(t)))

# 初始位置
q_actual[:, 0] = q0

# 仿真
for i in range(1, len(t)):
    # 当前末端位置和速度
    x = forward_kinematics(q_actual[:, i - 1])
    J = jacobian(q_actual[:, i - 1])
    dx = J @ dq_actual[:, i - 1]

    # 期望末端位置和速度
    x_d = forward_kinematics(qd)
    dx_d = np.zeros(3)

    # 计算期望末端力
    F_des = K_X @ (x_d - x) + K_B @ (dx_d - dx)

    # 计算控制力矩
    tau[:, i - 1] = J.T @ F_des

    # 在t=4到t=6之间施加外力
    if 4 <= t[i] <= 6:
        external_force = np.array([2, 3, 3])
        tau[:, i - 1] += external_force

    # 动力学模型（简化为单位惯性矩阵）
    ddq_actual[:, i - 1] = tau[:, i - 1]

    # 更新速度和位置
    dq_actual[:, i] = dq_actual[:, i - 1] + ddq_actual[:, i - 1] * dt
    q_actual[:, i] = q_actual[:, i - 1] + dq_actual[:, i - 1] * dt

# 最后一个时刻的控制力矩和加速度无法计算，赋值为最后一个已知值
tau[:, -1] = tau[:, -2]
ddq_actual[:, -1] = ddq_actual[:, -2]

# 绘制结果曲线
fig, axs = plt.subplots(4, 1, figsize=(10, 12))

# 关节位置
for i in range(3):
    axs[0].plot(t, q_actual[i], label=f'Joint {i + 1}')
axs[0].set_ylabel('position (rad or m)')
axs[0].set_title('joint_position')
axs[0].grid(True)
axs[0].legend()

# 关节速度
for i in range(3):
    axs[1].plot(t, dq_actual[i], label=f'Joint {i + 1}')
axs[1].set_ylabel('velocity (rad/s or m/s)')
axs[1].set_title('joint_velocity')
axs[1].grid(True)
axs[1].legend()

# 关节加速度
for i in range(3):
    axs[2].plot(t, ddq_actual[i], label=f'Joint {i + 1}')
axs[2].set_ylabel('accelaration (rad/s^2 or m/s^2)')
axs[2].set_title('joint_accelaration')
axs[2].grid(True)
axs[2].legend()

# 控制力矩
for i in range(3):
    axs[3].plot(t, tau[i], label=f'Joint {i + 1}')
axs[3].set_ylabel('control moment (N or Nm)')
axs[3].set_title('control moment')
axs[3].set_xlabel('time (s)')
axs[3].grid(True)
axs[3].legend()

plt.tight_layout()
plt.show()
