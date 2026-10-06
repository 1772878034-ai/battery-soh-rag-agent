import argparse
import json

from agent import Agent


def print_result(result):
    print("\n" + "=" * 70)
    print(f"[任务规划] 规划器来源: {result['planner_source']}")
    for i, step in enumerate(result["plan"], 1):
        print(f"  {i}. {step['tool']}({json.dumps(step.get('args', {}), ensure_ascii=False)})")
        print(f"     理由: {step.get('reason', '')}")

    print("\n[工具调用]")
    for call in result["tool_calls"]:
        print(f"  -> {call['tool']} {json.dumps(call['args'], ensure_ascii=False)}  ({call['latency_ms']} ms)")

    print("\n[回答]")
    print(result["answer"])

    if result["evidence"]:
        print("\n[证据展示]")
        for i, e in enumerate(result["evidence"], 1):
            print(f"  [{i}] cid={e['cid']} 重排得分={e['rerank_score']}")
            print(f"      {e['text']}")

    print("\n" + "=" * 70)
    print(f"总耗时: {result['timings']['total_ms']} ms")


def main():
    parser = argparse.ArgumentParser(description="锂电池 RAG 最小 Agent：工具调用 + 记忆 + 任务规划")
    parser.add_argument("question", nargs="?", help="问题；不传则进入交互模式")
    parser.add_argument("--session", default="cli", help="会话 id，记忆按会话隔离")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    args = parser.parse_args()

    agent = Agent(session_id=args.session)

    if args.question:
        result = agent.run(args.question)
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else print_result(result))
        return

    print("交互模式：输入问题回车；命令 /reset 清空记忆，/quit 退出")
    while True:
        try:
            question = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question in ("/quit", "/exit"):
            break
        if question == "/reset":
            agent.memory.reset()
            print("记忆已清空")
            continue
        result = agent.run(question)
        print_result(result)


if __name__ == "__main__":
    main()
