import ast
import operator
import time
from functools import lru_cache

import numpy as np
import scipy.io
from sklearn.preprocessing import MinMaxScaler

from . import config
from .retriever import get_retriever


class Tool:
    def __init__(self, name, description, func):
        self.name = name
        self.description = description
        self.func = func

    def run(self, **kwargs):
        return self.func(**kwargs)


REGISTRY = {}


def register(name, description):
    def decorator(func):
        REGISTRY[name] = Tool(name, description, func)
        return func

    return decorator


@register(
    "knowledge_search",
    "检索锂电池领域知识库，用于回答概念、定义、机理、原理类问题。"
    "参数：query(检索问题字符串)，top_k(返回片段数，默认3)。",
)
def knowledge_search(query, top_k=3):
    result = get_retriever().search(query, top_k=int(top_k))

    def pack(item):
        return {
            "cid": item["doc"].metadata["cid"],
            "ensemble_rank": item["ensemble_rank"],
            "rerank_score": item["rerank_score"],
            "text": item["doc"].page_content,
        }

    return {
        "query": query,
        "reranker": result["reranker"],
        "evidence": [pack(item) for item in result["ranked"]],
        "candidates": [pack(item) for item in result["candidates"]],
        "timings": result["timings"],
    }


@lru_cache(maxsize=1)
def _discharge_series():
    """读取 B0005，电流积分计算每个放电循环容量，返回 (容量序列, 原始循环编号)。"""
    mat = scipy.io.loadmat(str(config.MAT_PATH))
    cycles = mat["B0005"][0][0]["cycle"][0]
    capacities, raw_ids = [], []
    for i, cyc in enumerate(cycles):
        if cyc["type"][0] != "discharge":
            continue
        t = cyc["data"][0][0]["Time"][0]
        current = cyc["data"][0][0]["Current_measured"][0]
        dt = np.diff(t)
        capacities.append(float(np.sum(-current[1:] * dt) / 3600))
        raw_ids.append(i)
    return np.array(capacities), np.array(raw_ids)


@lru_cache(maxsize=1)
def _lstm_prediction():
    """与 day2 训练口径一致：全序列归一化、time_step=5，返回逐点预测容量。"""
    from tensorflow.keras.models import load_model

    capacities, _ = _discharge_series()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(capacities.reshape(-1, 1))
    time_step = 5
    X = []
    for i in range(time_step, len(scaled)):
        X.append(scaled[i - time_step : i, 0])
    X = np.reshape(np.array(X), (len(X), time_step, 1))
    model = load_model(str(config.LSTM_MODEL_PATH), compile=False)
    pred = scaler.inverse_transform(model.predict(X, verbose=0)).flatten()
    return pred, time_step, float(capacities[0])


@register(
    "predict_soh",
    "查询电池在指定循环次数附近的实测与LSTM预测放电容量(Ah)，用于数值类/趋势类问题。"
    "参数：cycle(整数循环次数)，rated_ah(额定容量Ah，默认2.0)。",
)
def predict_soh(cycle, rated_ah=config.RATED_CAPACITY_AH):
    t0 = time.perf_counter()
    cycle = int(cycle)
    capacities, raw_ids = _discharge_series()
    pred, time_step, initial = _lstm_prediction()

    ordinal = int(np.argmin(np.abs(raw_ids - cycle)))
    matched_cycle = int(raw_ids[ordinal])
    measured = round(float(capacities[ordinal]), 4)
    if ordinal >= time_step:
        predicted = round(float(pred[ordinal - time_step]), 4)
    else:
        predicted = measured

    return {
        "requested_cycle": cycle,
        "matched_cycle": matched_cycle,
        "measured_capacity_ah": measured,
        "predicted_capacity_ah": predicted,
        "initial_capacity_ah": round(initial, 4),
        "rated_capacity_ah": float(rated_ah),
        "soh_vs_rated_percent": round(predicted / float(rated_ah) * 100, 2),
        "retention_vs_initial_percent": round(predicted / initial * 100, 2),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("只支持数字与 + - * / % ** 运算")


@register(
    "calculator",
    "安全四则运算计算器，用于容量比值、百分比、下降幅度等数值计算。"
    "参数：expression(字符串，例如 '1.52/2.0*100')。",
)
def calculator(expression):
    value = _safe_eval(ast.parse(str(expression), mode="eval"))
    return {"expression": str(expression), "result": round(float(value), 4)}
