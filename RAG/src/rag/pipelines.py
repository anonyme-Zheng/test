from __future__ import annotations
from typing import List
import numpy as np

from .config import RAGConfig
from .embedder import SentenceTransformerEmbedder
from .indexer import ESIndexer
from .retriever import HybridRetriever
from .data_ingest import iterate_raw_files
from .generator import AnswerGenerator
from .query_rewrite import QueryRewriter
from .utils.io import logger

class RAGPipeline:
    def __init__(self, cfg:RAGConfig):
        self.cfg = cfg
        self.embedder = SentenceTransformerEmbedder(cfg.embedder)
        self.indexer  = ESIndexer(cfg.es)
        self.retriever= HybridRetriever(cfg.es)
        self.generator= AnswerGenerator(cfg.generator)
        self.rewriter = QueryRewriter(cfg.generator.llm_endpoint)

    # -------- offline ----------
    def build_index(self, source_dir:str):
        chunks=[]
        for ch in iterate_raw_files(source_dir):
            # TODO: 这里需要补充处理逻辑
            pass