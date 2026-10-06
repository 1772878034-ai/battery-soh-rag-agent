import os
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from . import config

# 逻辑核全开会过度争抢，限到物理核量级
torch.set_num_threads(max(1, min((os.cpu_count() or 8) // 2, 16)))


@lru_cache(maxsize=2)
def load_llm(model_name=None):
    """加载本地缓存的指令模型，返回 (HuggingFacePipeline 风格对象, tokenizer)。"""
    model_name = model_name or config.DEFAULT_LLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    except (ValueError, OSError):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to("cuda" if torch.cuda.is_available() else "cpu")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
        return_full_text=False,
        repetition_penalty=1.05,
    )
    return pipe, tokenizer


def chat_prompt(tokenizer, system, user, history=None):
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(system, user, history=None, model_name=None, max_new_tokens=512,
             temperature=0.2, do_sample=None):
    """统一的对话生成入口，返回纯文本。temperature=0 时默认贪心解码。"""
    pipe, tokenizer = load_llm(model_name)
    prompt = chat_prompt(tokenizer, system, user, history)
    if do_sample is None:
        do_sample = temperature > 0
    output = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        return_full_text=False,
    )
    return output[0]["generated_text"].strip()
