import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# 解决中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ==========1.读取NASA电池数据，和day1代码一致==========
mat = scipy.io.loadmat("B0005.mat")
data = mat['B0005'][0][0]
cycles = data['cycle'][0]

cap_list = []
cycle_id = []
for i, cyc in enumerate(cycles):
    cyc_type = cyc['type'][0]
    if cyc_type == 'discharge':
        t = cyc['data'][0][0]['Time'][0]
        I = cyc['data'][0][0]['Current_measured'][0]
        dt = np.diff(t)
        cap = np.sum(-I[1:] * dt) / 3600
        cap_list.append(cap)
        cycle_id.append(i)

cap_array = np.array(cap_list).reshape(-1,1)

# ==========2.数据归一化==========
scaler = MinMaxScaler(feature_range=(0,1))
cap_scaled = scaler.fit_transform(cap_array)

# ==========3.构造时序样本：用前5个循环预测第6个==========
time_step = 5
X, y = [], []
for i in range(time_step, len(cap_scaled)):
    X.append(cap_scaled[i-time_step:i,0])
    y.append(cap_scaled[i,0])
X, y = np.array(X), np.array(y)

# LSTM输入形状：[样本数, 时间步, 特征维度]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# ==========4.搭建LSTM模型==========
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1],1)))
model.add(Dense(units=25))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')


# ==========5.训练模型==========
print("开始训练LSTM模型...")
model.fit(X, y, batch_size=8, epochs=30)
model.save("lstm_battery_model.h5")

# ==========6.预测+反归一化==========
y_pred_scaled = model.predict(X)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y.reshape(-1,1))

# ==========7.绘图对比真实值与预测值==========
plt.figure(figsize=(10,4))
plt.plot(y_true, label="真实容量", color="#1f77b4")
plt.plot(y_pred, label="LSTM预测容量", color="#ff7f0e")
plt.xlabel("循环序号")
plt.ylabel("容量(Ah)")
plt.title("NASA B0005 LSTM电池容量预测")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("lstm_cap_pred.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅Day2运行完成，图片lstm_cap_pred.png已保存")