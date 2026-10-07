"""端到端入口：加载 -> 划分/归一化 -> 开窗 -> 训练 -> 评估 SOH -> 绘图。

在仓库根目录运行：
    python run_pipeline.py
"""

import os
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.battery_pipeline import DataLoader, FeatureBuilder, ModelRunner

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 固定随机种子，保证结果可复现
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "B0005.mat")
MODEL_PATH = os.path.join(BASE_DIR, "data", "lstm_battery_model.keras")
FIG_PATH = os.path.join(BASE_DIR, "data", "lstm_soh_pred.png")

TIME_STEP = 5
NOMINAL_CAPACITY = 2.0        # NASA 18650 额定容量(Ah)
SCALE_BOUNDS = (50.0, 105.0)  # SOH 固定物理量程(%)，覆盖全寿命区间


def main():
    # 1. 读取容量并换算成 SOH
    loader = DataLoader(DATA_PATH, nominal_capacity=NOMINAL_CAPACITY,
                        train_ratio=0.7, val_ratio=0.15, scale_bounds=SCALE_BOUNDS)
    capacity = loader.load_capacities()
    soh = loader.capacity_to_soh(capacity)
    n = len(soh)
    train_end, val_end = loader.split_points(n)
    print(f"放电循环数: {n}")
    print(f"时间分界: train <{train_end} | val [{train_end},{val_end}) | test >={val_end}")

    # 2. 确定缩放基准（固定物理量程），再缩放整条序列
    loader.fit_scaler(soh[:train_end])
    soh_scaled = loader.scale(soh)

    # 3. 构造滑动窗口样本
    builder = FeatureBuilder(time_step=TIME_STEP)
    sets = builder.build(soh_scaled, train_end, val_end)
    X_train, y_train, _ = sets["train"]
    X_val, y_val, _ = sets["val"]
    X_test, y_test, _ = sets["test"]
    print(f"窗口样本数 train/val/test = {len(y_train)}/{len(y_val)}/{len(y_test)}")

    # 4. 搭建并训练（验证集 + EarlyStopping）
    runner = ModelRunner(time_step=TIME_STEP).build()
    runner.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=8)
    runner.save(MODEL_PATH)

    # 5. 各集合预测、反归一化并计算 SOH 误差
    results = {}
    for name in ("train", "val", "test"):
        X, y_scaled, idx = sets[name]
        pred = loader.inverse_scale(runner.predict(X))
        true = loader.inverse_scale(y_scaled)
        metrics = ModelRunner.evaluate(true, pred)
        results[name] = (idx, true, pred)
        print(f"[{name:>5}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    # 6. 绘图：真实 SOH 曲线 + 各集合预测点 + 分界虚线
    colors = {"train": "#1f77b4", "val": "#2ca02c", "test": "#d62728"}
    plt.figure(figsize=(11, 5))
    plt.plot(soh, color="#999999", linewidth=2, label="真实 SOH")
    for name in ("train", "val", "test"):
        idx, _, pred = results[name]
        plt.plot(idx, pred, "o", markersize=3.5, color=colors[name],
                 label=f"{name} 预测")
    for boundary in (train_end, val_end):
        plt.axvline(boundary, linestyle="--", color="#555555", linewidth=1)
    plt.xlabel("放电循环序号")
    plt.ylabel("SOH (%)")
    plt.title("NASA B0005 LSTM 的 SOH 预测（时间顺序 train/val/test）")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    print(f"模型已保存: {MODEL_PATH}")
    print(f"对比图已保存: {FIG_PATH}")


if __name__ == "__main__":
    main()
