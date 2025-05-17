from __future__ import annotations
import requests
from typing import List
from .config import GeneratorConfig
from .utils.io import logger

class AnswerGenerator:
    def __init__(self,cfg:GeneratorConfig):
        self.cfg = cfg

    def generate(self, query:str, contexts:List[dict])->str:
        context_text = "\n".join(f"[{c['score']:.2f}] {c['text']}" for c in contexts)
        messages=[
            {"role":"system","content":self.cfg.system_prompt},
            {"role":"user","content":f"参考以下资料回答：\n{context_text}\n\n问题:{query}"}
        ]
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens
        }
        resp = requests.post(self.cfg.llm_endpoint, json=payload, timeout=60).json()
        answer = resp["choices"][0]["message"]["content"].strip()
        logger.info("LLM answer tokens: %d", resp["usage"]["completion_tokens"])
        return answer