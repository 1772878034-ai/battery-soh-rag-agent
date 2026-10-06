import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# 解决中文乱码
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

mat = scipy.io.loadmat("B0005.mat")
data = mat['B0005'][0][0]
cycles = data['cycle'][0]

cap_list = []
cycle_id = []

for i, cyc in enumerate(cycles):
    cyc_type = cyc['type'][0]
    if cyc_type == 'discharge':
        # 正确字段名：Current_measured
        t = cyc['data'][0][0]['Time'][0]
        I = cyc['data'][0][0]['Current_measured'][0]
        # 电流积分求容量
        dt = np.diff(t)
        cap = np.sum(-I[1:] * dt) / 3600
        cap_list.append(cap)
        cycle_id.append(i)

plt.figure(figsize=(10,4))
plt.plot(cycle_id, cap_list, color="#1f77b4", linewidth=2)
plt.xlabel("Cycle Number 循环次数")
plt.ylabel("Capacity 容量(Ah)")
plt.title("NASA B0005 电池容量衰减曲线")
plt.grid(True, alpha=0.3)
plt.savefig("capacity_curve.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"一共提取 {len(cap_list)} 组放电循环")
print("容量数组：", cap_list)