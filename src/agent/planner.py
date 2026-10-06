import json
import re

from . import config, llm
from .tools import REGISTRY

_TOOL_SPECS = "\n".join(f"- {name}: {tool.description}" for name, tool in REGISTRY.items())

PLANNER_SYSTEM = f"""你是锂电池问答助手的任务规划器。可用工具如下：
{_TOOL_SPECS}

规划规则：
1. 只输出严格 JSON，不要输出任何解释或 markdown 代码块，格式为
   {{"steps": [{{"tool": "工具名", "args": {{...}}, "reason": "一句话说明"}}]}}
2. 容量/SOH/预测/趋势类问题用 predict_soh；概念、定义、机理、原理类问题用 knowledge_search；
   百分比、比值等算术用 calculator。
3. 后一步可以引用前一步的结果，写法为 {{{{工具名.字段名}}}}，例如
   calculator 的 expression 可以写 "{{{{predict_soh.predicted_capacity_ah}}}}/2.0*100"。
4. 一次最多 {config.MAX_PLAN_STEPS} 个步骤，按执行顺序排列；只规划与问题直接相关的工具，
   不要为没有提问的内容安排工具。
示例：
用户：B0005电池循环到100次时容量还剩多少？SOH是多少？为什么会衰减？
输出：{{"steps": [{{"tool": "predict_soh", "args": {{"cycle": 100}}, "reason": "获取第100次循环的预测容量"}}, {{"tool": "calculator", "args": {{"expression": "{{{{predict_soh.predicted_capacity_ah}}}}/2.0*100"}}, "reason": "由预测容量计算SOH百分比"}}, {{"tool": "knowledge_search", "args": {{"query": "锂电池容量衰减机理 SEI膜 析锂"}}, "reason": "检索容量衰减机理资料"}}]}}"""

_FOLLOW_RE = re.compile(r"再(?:循环|往后|多)?\s*(\d+)\s*次")
_CYCLE_RE = re.compile(r"(?:循环到|到第?|第)\s*(\d+)\s*次")


def resolve_references(question, memory):
    """把“再循环50次”这类指代结合记忆解析成具体循环次数。"""
    hints = {}
    follow = _FOLLOW_RE.search(question)
    last_cycle = memory.get("last_cycle")
    if follow and last_cycle is not None:
        target = int(last_cycle) + int(follow.group(1))
        hints["cycle"] = target
        return f"{question}（即循环到第{target}次）", hints
    explicit = _CYCLE_RE.search(question)
    if explicit:
        hints["cycle"] = int(explicit.group(1))
    return question, hints


def _parse_plan(raw):
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("规划结果中没有 JSON")
    plan = json.loads(match.group(0))
    steps = plan["steps"]
    for step in steps:
        if step["tool"] not in REGISTRY:
            raise ValueError(f"未知工具: {step['tool']}")
        if not isinstance(step.get("args"), dict):
            raise ValueError("args 必须是对象")
    if not steps or len(steps) > config.MAX_PLAN_STEPS:
        raise ValueError("步骤数量不合法")
    return steps


def _rule_plan(question, hints):
    """LLM 规划失败时的确定性路由兜底。"""
    steps = []
    cycle = hints.get("cycle")
    asks_numeric = re.search(r"(容量|SOH|健康度|还剩|预测|趋势|多少)", question)
    asks_knowledge = re.search(r"(为什么|机理|是什么|什么是|定义|原理|SEI|析锂|BMS|作用|怎么)", question)

    if cycle is not None and asks_numeric:
        steps.append({"tool": "predict_soh", "args": {"cycle": cycle}, "reason": f"获取第{cycle}次循环容量"})
        if re.search(r"(SOH|健康度|百分比|比例|多少)", question):
            steps.append({
                "tool": "calculator",
                "args": {"expression": "{{predict_soh.predicted_capacity_ah}}/2.0*100"},
                "reason": "由预测容量计算 SOH 百分比",
            })
    if asks_knowledge:
        query = re.sub(r"[？?。，,\s]+", "", question)[-30:]
        steps.append({"tool": "knowledge_search", "args": {"query": query}, "reason": "检索领域知识库"})
    if not steps:
        steps.append({"tool": "knowledge_search", "args": {"query": question}, "reason": "默认走知识库检索"})
    return steps[: config.MAX_PLAN_STEPS]


def plan(question, memory):
    resolved, hints = resolve_references(question, memory)
    try:
        raw = llm.generate(
            PLANNER_SYSTEM, resolved, max_new_tokens=256, temperature=0.0, do_sample=False
        )
        steps = _parse_plan(raw)
        return resolved, hints, steps, "llm"
    except Exception:
        return resolved, hints, _rule_plan(resolved, hints), "router"
