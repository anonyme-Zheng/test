from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml, os

@dataclass(slots=True)
class ESConfig:
    host: str = "http://localhost:9200"
    index_name: str = "financial_report_data"
    embedding_dim: int = 768
    username: str | None = None
    password: str | None = None

@dataclass(slots=True)
class EmbedderConfig:
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    batch_size: int = 32
    device: str = "cuda"

@dataclass(slots=True)
class GeneratorConfig:
    llm_endpoint: str = "http://localhost:8000/v1/chat/completions"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 512
    system_prompt: str = """你是一位金融分析助手，回答需引用检索结果并用中文输出。"""

@dataclass(slots=True)
class RAGConfig:
    es: ESConfig = field(default_factory=ESConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)

    @staticmethod
    def from_yaml(path:str|os.PathLike) -> 'RAGConfig':
        with open(path,'r',encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return RAGConfig(**data)
