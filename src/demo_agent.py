"""1 分钟端到端演示：问题输入 → 任务规划 → 混合检索 → BGE 重排 → 工具调用 → 回答 → 证据展示。

启动时先一次性预热全部模型（约 1-2 分钟），预热不计入演示计时；
演示实录自动写入 docs/demo_transcript.md。
"""
import time

from agent import Agent
from agent import config

QUESTIONS = [
    "B0005电池循环到100次时容量大概还剩多少？SOH是多少？另外说一下容量为什么会衰减。",
    "那再循环50次呢？容量和SOH变成多少？",
]


class Recorder:
    def __init__(self):
        self.lines = []

    def emit(self, text=""):
        print(text)
        self.lines.append(text)

    def rule(self, title):
        self.emit("")
        self.emit("=" * 78)
        self.emit(title)
        self.emit("=" * 78)


def warmup(rec):
    rec.rule("预热阶段（一次性，不计入演示计时）")
    rec.emit("加载 LLM ...")
    from agent.llm import load_llm

    load_llm()
    rec.emit("加载混合检索器与 BGE 重排器 ...")
    from agent.retriever import get_retriever

    result = get_retriever().search("预热")
    rec.emit(f"  重排器: {result['reranker']}")
    rec.emit("加载 LSTM 并跑通一次预测（首次会加载 TensorFlow）...")
    from agent.tools import predict_soh

    probe = predict_soh(50)
    rec.emit(f"  探针结果: 第{probe['matched_cycle']}次循环预测容量 {probe['predicted_capacity_ah']} Ah")


def show_turn(rec, idx, question, result):
    rec.rule(f"演示第 {idx} 轮 | 步骤1：问题输入")
    rec.emit(f"用户问题：{question}")
    if result["resolved"] != question:
        rec.emit(f"记忆指代解析后：{result['resolved']}")

    rec.emit("")
    rec.emit("步骤2：任务规划（LLM 输出工具序列，规划器来源: "
             f"{result['planner_source']}）")
    for i, step in enumerate(result["plan"], 1):
        args = ", ".join(f"{k}={v!r}" for k, v in step.get("args", {}).items())
        rec.emit(f"  {i}. {step['tool']}({args})")
        rec.emit(f"     理由: {step.get('reason', '')}")

    rec.emit("")
    rec.emit("步骤3/4：混合检索（BM25+FAISS RRF）→ BGE 重排")
    for call in result["tool_calls"]:
        if call["tool"] != "knowledge_search":
            continue
        output = call["output"]
        rec.emit(f"  检索问题: {output['query']}    重排器: {output['reranker']}")
        rec.emit("  融合召回候选（按重排得分排序，* 为最终采用片段）:")
        for j, cand in enumerate(output.get("candidates", []), 1):
            used = "*" if j <= len(output["evidence"]) else " "
            rec.emit(
                f"   {used} 重排后第{j} | cid={cand['cid']} | "
                f"融合阶段排名={cand['ensemble_rank']} | 重排得分={cand['rerank_score']}"
            )
        rec.emit(f"  检索耗时: 召回 {output['timings']['recall_ms']} ms / "
                 f"重排 {output['timings']['rerank_ms']} ms")

    rec.emit("")
    rec.emit("步骤5：工具调用执行")
    for call in result["tool_calls"]:
        args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
        rec.emit(f"  -> {call['tool']}({args})  [{call['latency_ms']} ms]")
        if call["tool"] == "predict_soh":
            out = call["output"]
            rec.emit(
                f"     匹配第{out['matched_cycle']}次循环 | "
                f"实测 {out['measured_capacity_ah']} Ah | "
                f"LSTM预测 {out['predicted_capacity_ah']} Ah | "
                f"SOH {out['soh_vs_rated_percent']}%"
            )
        if call["tool"] == "calculator":
            out = call["output"]
            rec.emit(f"     {out['expression']} = {out['result']}")

    rec.emit("")
    rec.emit("步骤6：生成回答")
    rec.emit(result["answer"])

    rec.emit("")
    rec.emit("步骤7：证据展示")
    if result["evidence"]:
        for i, e in enumerate(result["evidence"], 1):
            rec.emit(f"  [{i}] cid={e['cid']} | 重排得分={e['rerank_score']} | "
                     f"来源: data/battery_info.txt")
            rec.emit(f"      {e['text']}")
    else:
        rec.emit("  本轮为数值工具调用，证据为 B0005 实测数据与 LSTM 输出，见步骤5。")


def main():
    rec = Recorder()
    rec.emit("锂电池 RAG Agent 一分钟演示")
    rec.emit(f"LLM: {config.DEFAULT_LLM} | 嵌入: {config.EMBED_MODEL} | "
             f"重排: {config.RERANK_MODEL}")

    warmup(rec)

    agent = Agent(session_id="demo1min")
    agent.memory.reset()

    rec.rule("正式演示开始计时")
    t0 = time.perf_counter()
    results = []
    for i, question in enumerate(QUESTIONS, 1):
        turn_t0 = time.perf_counter()
        result = agent.run(question)
        result["wall_ms"] = round((time.perf_counter() - turn_t0) * 1000, 1)
        results.append(result)
        show_turn(rec, i, question, result)
        rec.emit(f"  本轮耗时: {result['wall_ms']} ms")

    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    rec.rule("演示结束")
    rec.emit(f"两轮问答正式演示总耗时: {total_ms} ms（预热未计入）")

    docs_dir = config.PROJECT_ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    transcript = docs_dir / "demo_transcript.md"
    body = [
        "# 1 分钟演示实录",
        "",
        f"- LLM: `{config.DEFAULT_LLM}`；嵌入: `{config.EMBED_MODEL}`；重排: `{config.RERANK_MODEL}`",
        "- 模型预热为一次性开销，未计入演示耗时。",
        "",
        "```text",
        "\n".join(rec.lines),
        "```",
        "",
    ]
    transcript.write_text("\n".join(body), encoding="utf-8")
    print(f"\n实录已写入: {transcript}")


if __name__ == "__main__":
    main()
