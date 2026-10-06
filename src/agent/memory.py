import json

from . import config


class ConversationMemory:
    """短期对话窗口 + 长期事实槽，JSON 落盘，跨进程保留。"""

    def __init__(self, session_id="default", window_turns=6):
        self.session_id = session_id
        self.window_turns = window_turns
        self.path = config.STATE_DIR / f"memory_{session_id}.json"
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.turns = data.get("turns", [])
            self.facts = data.get("facts", {})
        else:
            self.turns = []
            self.facts = {}

    def add_user(self, content):
        self.turns.append({"role": "user", "content": content})

    def add_assistant(self, content):
        self.turns.append({"role": "assistant", "content": content})

    def history_window(self):
        return self.turns[-self.window_turns :]

    def remember(self, key, value):
        self.facts[key] = value

    def get(self, key, default=None):
        return self.facts.get(key, default)

    def save(self):
        self.path.write_text(
            json.dumps({"turns": self.turns[-50:], "facts": self.facts}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def reset(self):
        self.turns = []
        self.facts = {}
        self.save()
