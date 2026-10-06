import json
import re
import time

from . import config, llm
from .memory import ConversationMemory
from .planner import plan as make_plan
from .tools import REGISTRY

PROSE_SYSTEM = (
    "你是锂电池领域专家。只能依据用户给出的知识库片段回答，"
    "片段中没有的内容必须明确说“知识库暂无相关信息”，禁止编造。\n"
    "输出格式：直接输出一个分点列表，只保留这一个列表，"
    "不要设置多个小节或小标题，不要输出列表之外的任何内容，不要复述问题。\n"
    "内容要求：问到机理/原因时，把片段中的机理条目逐条列出"
    "（如 SEI 膜持续生长、活性材料结构破损脱落、析锂）及加速老化的因素；"
    "在引用句末标注编号，如[1]；不要重复已经给出的数值结论，"
    "同一条目只出现一次，总字数不超过 160 字。"
)

_TEMPLATE_RE = re.compile(r"\{\{\s*(\w+)\.([\w]+)\s*\}\}")

_NUMERIC_INTENT = re.compile(r"(容量|SOH|健康度|还剩|预测|趋势|多少)")
_PROSE_INTENT = re.compile(r"(为什么|机理|是什么|什么是|定义|原理|解释|啥是)")


def prune_plan(steps, question):
    """按问题意图裁剪 LLM 规划：没问的不调用；calculator 依赖 predict。"""
    wants_numeric = bool(_NUMERIC_INTENT.search(question))
    wants_prose = bool(_PROSE_INTENT.search(question))
    kept = []
    for step in steps:
        tool = step["tool"]
        if tool in ("predict_soh", "calculator") and not wants_numeric:
            continue
        if tool == "knowledge_search" and not wants_prose:
            continue
        kept.append(step)
    if not any(s["tool"] == "predict_soh" for s in kept):
        kept = [s for s in kept if s["tool"] != "calculator"]
    if not kept:
        kept = [{"tool": "knowledge_search", "args": {"query": question}, "reason": "默认走知识库检索"}]
    return kept


class Agent:
    """规划 → 工具调用 → 综合回答 的最小 Agent，带会话记忆。"""

    def __init__(self, session_id="default"):
        self.session_id = session_id
        self.memory = ConversationMemory(session_id)

    @staticmethod
    def _render_args(args, observations):
        rendered = {}
        for key, value in args.items():
            if isinstance(value, str):
                value = _TEMPLATE_RE.sub(
                    lambda m: str(observations.get(m.group(1), {}).get(m.group(2), m.group(0))),
                    value,
                )
            rendered[key] = value
        return rendered

    @staticmethod
    def _numeric_sentence(out):
        return (
            f"数值结论：在第{out['matched_cycle']}次循环附近"
            f"（请求第{out['requested_cycle']}次），实测放电容量 {out['measured_capacity_ah']} Ah，"
            f"LSTM 预测容量 {out['predicted_capacity_ah']} Ah；"
            f"按额定容量 {out['rated_capacity_ah']} Ah 计，SOH ≈ {out['soh_vs_rated_percent']}%，"
            f"相对初始容量 {out['initial_capacity_ah']} Ah 的保持率为 "
            f"{out['retention_vs_initial_percent']}%。"
        )

    @staticmethod
    def _template_answer(calls, evidence):
        lines = ["（LLM 暂不可用，以下为工具结果直出）"]
        for call in calls:
            lines.append(f"- {call['tool']}: {json.dumps(call['output'], ensure_ascii=False)}")
        for i, e in enumerate(evidence, 1):
            lines.append(f"[{i}] {e['text']}")
        return "\n".join(lines)

    def run(self, question):
        t_start = time.perf_counter()
        resolved, hints, steps, planner_source = make_plan(question, self.memory)
        steps = prune_plan(steps, resolved)
        t_planned = time.perf_counter()

        calls, observations, evidence = [], {}, []
        for step in steps:
            tool = REGISTRY[step["tool"]]
            args = self._render_args(step.get("args", {}), observations)
            t0 = time.perf_counter()
            try:
                output = tool.run(**args)
            except Exception as exc:
                output = {"error": f"{type(exc).__name__}: {exc}"}
            latency = round((time.perf_counter() - t0) * 1000, 1)
            calls.append({"tool": tool.name, "args": args, "output": output, "latency_ms": latency})
            observations.setdefault(tool.name, {}).update(output)
            if tool.name == "knowledge_search":
                evidence.extend(output.get("evidence", []))
        t_tools_done = time.perf_counter()

        answer_parts = []
        for call in calls:
            out = call["output"]
            if call["tool"] == "predict_soh" and "error" not in out:
                answer_parts.append(self._numeric_sentence(out))
            if call["tool"] == "calculator" and "error" not in out:
                answer_parts.append(f"计算核验：{out['expression']} = {out['result']}")

        wants_prose = bool(_PROSE_INTENT.search(resolved))
        if evidence and wants_prose:
            evidence_blocks = [
                f"[{i}]（cid={e['cid']}）{e['text']}" for i, e in enumerate(evidence, 1)
            ]
            prose_user = f"用户问题：{resolved}\n\n知识库片段：\n" + "\n".join(evidence_blocks)
            try:
                prose = llm.generate(
                    PROSE_SYSTEM,
                    prose_user,
                    max_new_tokens=160,
                    temperature=0.0,
                    do_sample=False,
                )
                answer_parts.append(prose)
            except Exception:
                answer_parts.append(self._template_answer(calls, evidence))
        elif any(call["tool"] == "knowledge_search" for call in calls):
            answer_parts.append("知识库暂无相关信息。")

        answer = "\n\n".join(answer_parts)
        t_answer_done = time.perf_counter()

        total_ms = round((t_answer_done - t_start) * 1000, 1)
        self.memory.add_user(question)
        self.memory.add_assistant(answer)
        if hints.get("cycle") is not None:
            self.memory.remember("last_cycle", hints["cycle"])
        self.memory.remember("battery_id", "B0005")
        self.memory.save()

        return {
            "question": question,
            "resolved": resolved,
            "planner_source": planner_source,
            "plan": steps,
            "tool_calls": calls,
            "evidence": evidence,
            "answer": answer,
            "timings": {
                "total_ms": total_ms,
                "planning_ms": round((t_planned - t_start) * 1000, 1),
                "tools_ms": round((t_tools_done - t_planned) * 1000, 1),
                "answer_ms": round((t_answer_done - t_tools_done) * 1000, 1),
            },
        }
