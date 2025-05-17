from __future__ import annotations
from typing import List
import requests, json

_REWRITE_PROMPT = """根据上下文历史对话对 query 改写，进行补全/指代，关键词提取：
历史对话:
{history}
用户提问:
{query}
请输出改写后的完整查询:"""

class QueryRewriter:
    def __init__(self,llm_endpoint:str):
        self.url = llm_endpoint

    def rewrite(self, history:List[str], query:str)->dict:
        prompt = _REWRITE_PROMPT.format(history="\n".join(history[-5:]), query=query)
        payload = {
            "model":"gpt-4o-mini",
            "messages":[{"role":"user","content":prompt}],
            "temperature":0.0,
            "max_tokens":128,
        }
        resp = requests.post(self.url, json=payload, timeout=20).json()
        content = resp["choices"][0]["message"]["content"].strip()
        return {"rewritten": content, "keywords": []}
