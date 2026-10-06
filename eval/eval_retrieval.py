"""检索评估：BM25 / FAISS / 混合RRF / 混合+BGE重排 四种方案对比。

指标：Hit@k、Recall@k、MRR（k=1,2,3），以及平均单问延迟。
标注方式：每个问题给出 gold_terms（话题 chunk 中必须同时出现的关键短语），
运行时自动映射到 chunk id，避免硬编码下标。

运行：python eval/eval_retrieval.py
输出：eval/results/retrieval_eval.json 与 retrieval_eval.md
"""
import json
import os
import sys
import time
from pathlib import Path

# 必须在任何 huggingface_hub/transformers 导入之前设置，否则离线开关不会生效
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_community.retrievers import BM25Retriever  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_classic.retrievers import EnsembleRetriever  # noqa: E402

from agent.retriever import _cross_encoder, embeddings, load_chunks  # noqa: E402
from agent import config  # noqa: E402

# (问题, gold_terms)；chunk 同时包含全部 term 即视为相关
LABELED = [
    ("什么是电池的SOH？", ["SOH定义为"]),
    ("SOH是怎么定义的？", ["SOH定义为"]),
    ("电池健康状态这个指标是什么？", ["SOH定义为"]),
    ("电池老化之后会有什么表现？", ["可用容量下降", "内阻上升"]),
    ("电池容量衰减的机理有哪些？", ["容量衰减机理"]),
    ("SEI膜持续生长会造成什么影响？", ["SEI", "消耗活性锂"]),
    ("什么是析锂现象？", ["析锂"]),
    ("哪些因素会加速电池老化？", ["充放电倍率", "环境温度", "截止电压"]),
    ("电池用久了容量为什么会掉？", ["容量衰减机理"]),
    ("NASA锂电池数据集记录了哪些数据？", ["NASA", "电压", "电流"]),
    ("有没有公开的电池循环老化实验数据？", ["NASA", "充放电实验数据"]),
    ("做电池SOH预测算法研究一般用什么数据集？", ["NASA", "SOH预测算法研究"]),
    ("为什么说LSTM适合做电池容量预测？", ["LSTM", "长期依赖"]),
    ("LSTM是怎么实现电池健康提前预警的？", ["LSTM", "提前预警"]),
    ("怎样才能捕捉容量随循环变化的长期依赖？", ["LSTM", "长期依赖"]),
    ("RAG检索增强生成技术有什么作用？", ["RAG", "知识库"]),
    ("怎么做才能减少大模型的幻觉？", ["减少模型幻觉"]),
    ("为什么要把领域专业文献做成知识库？", ["RAG", "知识库"]),
]

KS = (1, 2, 3)


def gold_cids(chunks, terms):
    ids = []
    for chunk in chunks:
        if all(term in chunk.page_content for term in terms):
            ids.append(chunk.metadata["cid"])
    if not ids:
        raise ValueError(f"标注在知识库中找不到对应 chunk: {terms}")
    return ids


def metrics_for(ranking, gold):
    row = {}
    for k in KS:
        top = ranking[:k]
        row[f"hit@{k}"] = float(any(g in top for g in gold))
        row[f"recall@{k}"] = len(set(top) & set(gold)) / len(gold)
    row["mrr"] = next((1.0 / (i + 1) for i, c in enumerate(ranking) if c in gold), 0.0)
    return row


def main():
    chunks = load_chunks()
    print(f"知识库 chunk 数: {len(chunks)}")

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5
    faiss_db = FAISS.load_local(
        str(config.FAISS_DIR), embeddings(), allow_dangerous_deserialization=True
    )
    faiss_r = faiss_db.as_retriever(search_kwargs={"k": 5})
    ensemble = EnsembleRetriever(retrievers=[bm25, faiss_r], weights=[0.5, 0.5])
    reranker = _cross_encoder()

    def rankings(query):
        bm = BM25Retriever.from_documents(chunks)
        bm.k = 5
        bm_list = [d.metadata["cid"] for d in bm.invoke(query)]
        faiss_list = [
            d.metadata.get("cid", -1) for d in faiss_db.similarity_search(query, k=5)
        ]
        ens_docs = ensemble.invoke(query)
        ens_list = [d.metadata.get("cid", -1) for d in ens_docs]
        scores = reranker.predict([(query, d.page_content) for d in ens_docs])
        rerank_docs = [d for _, d in sorted(zip(scores, ens_docs), key=lambda x: x[0], reverse=True)]
        rerank_list = [d.metadata.get("cid", -1) for d in rerank_docs]
        return {
            "bm25": bm_list,
            "faiss": faiss_list,
            "ensemble": ens_list,
            "rerank": rerank_list,
        }

    # 预热一次（模型/检索器首次加载不计入延迟）
    rankings(LABELED[0][0])

    per_question, latency_sum, latency_n = [], {k: 0.0 for k in ("bm25", "faiss", "ensemble", "rerank")}, 0
    PASSES = 3
    for _ in range(PASSES):
        for qi, (question, terms) in enumerate(LABELED):
            gold = gold_cids(chunks, terms)
            t0 = time.perf_counter()
            r = rankings(question)
            per_variant_dt = {}
            # 分阶段计时（单独再测一遍各阶段）
            t = time.perf_counter()
            bm = BM25Retriever.from_documents(chunks)
            bm.k = 5
            bm.invoke(question)
            per_variant_dt["bm25"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            faiss_db.similarity_search(question, k=5)
            per_variant_dt["faiss"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            ens_docs = ensemble.invoke(question)
            per_variant_dt["ensemble"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            scores = reranker.predict([(question, d.page_content) for d in ens_docs])
            sorted(zip(scores, ens_docs), key=lambda x: x[0], reverse=True)
            per_variant_dt["rerank"] = (time.perf_counter() - t) * 1000
            for name, ms in per_variant_dt.items():
                latency_sum[name] += ms
            latency_n += 1

            if _ == 0:
                per_question.append({
                    "question": question,
                    "gold": gold,
                    "rankings": r,
                    "metrics": {name: metrics_for(rank, gold) for name, rank in r.items()},
                })

    variants = ["bm25", "faiss", "ensemble", "rerank"]
    aggregate = {}
    for name in variants:
        agg = {}
        for metric in (f"hit@{k}" for k in KS):
            agg[metric] = round(
                sum(q["metrics"][name][metric] for q in per_question) / len(per_question), 4
            )
        for metric in (f"recall@{k}" for k in KS):
            agg[metric] = round(
                sum(q["metrics"][name][metric] for q in per_question) / len(per_question), 4
            )
        agg["mrr"] = round(
            sum(q["metrics"][name]["mrr"] for q in per_question) / len(per_question), 4
        )
        agg["latency_ms"] = round(latency_sum[name] / latency_n, 1)
        aggregate[name] = agg

    results = {
        "n_questions": len(LABELED),
        "passes": PASSES,
        "n_chunks": len(chunks),
        "aggregate": aggregate,
        "per_question": per_question,
    }

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "retrieval_eval.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    headers = ["方案", "Hit@1", "Hit@2", "Hit@3", "Recall@1", "Recall@3", "MRR", "延迟ms"]
    lines = ["# 检索评估结果", "", f"问题数: {len(LABELED)}，每个方案跑 {PASSES} 轮，延迟为单问均值。", "",
             "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    labels = {"bm25": "BM25", "faiss": "FAISS 稠密", "ensemble": "混合 RRF", "rerank": "混合 + BGE 重排"}
    for name in variants:
        a = aggregate[name]
        lines.append("| " + " | ".join([
            labels[name], f"{a['hit@1']:.3f}", f"{a['hit@2']:.3f}", f"{a['hit@3']:.3f}",
            f"{a['recall@1']:.3f}", f"{a['recall@3']:.3f}", f"{a['mrr']:.3f}",
            f"{a['latency_ms']:.1f}",
        ]) + " |")
    (out_dir / "retrieval_eval.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\n已写入 {out_dir}")


if __name__ == "__main__":
    main()
