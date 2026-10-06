import os
from pathlib import Path

# 路径全部以项目根目录为基准，脚本在任意 cwd 下都能跑
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_PATH = DATA_DIR / "battery_info.txt"
FAISS_DIR = DATA_DIR / "faiss_db"
MAT_PATH = DATA_DIR / "B0005.mat"
LSTM_MODEL_PATH = DATA_DIR / "lstm_battery_model.h5"
LOCAL_MODEL_CACHE = DATA_DIR / "model"
STATE_DIR = DATA_DIR / "agent_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

# 演示与答辩环境默认离线，只用本机已缓存的模型
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# CPU 演示默认 0.5B（约 6 tok/s）；机器有 GPU 或追求质量时可设
# AGENT_LLM=Qwen/Qwen2.5-1.5B-Instruct
DEFAULT_LLM = os.environ.get("AGENT_LLM", "Qwen/Qwen2.5-0.5B-Instruct")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "BAAI/bge-reranker-base"

# NASA PCoE 18650 电池额定容量约 2.0 Ah；SOH 同时给出相对额定/初始容量两种口径
RATED_CAPACITY_AH = 2.0

MAX_PLAN_STEPS = 4
