from __future__ import annotations
from typing import List
import os
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
    def build_index(self, source_dir: str):
        """Ingest raw data and build the retrieval index."""
        chunks = list(iterate_raw_files(source_dir))
        if not chunks:
            logger.warning("No documents found under %s", source_dir)
            return

        texts = [c["text"] for c in chunks]
        vectors = np.array(self.embedder.encode(texts))
        self.indexer.index_chunks(chunks, vectors)

    # -------- online ----------
    def answer(self, query: str, history: List[str]) -> str:
        """Generate an answer for ``query`` given conversation ``history``."""
        rewritten = self.rewriter.rewrite(history, query).get("rewritten", query)
        query_vec = np.array(self.embedder.encode([rewritten]))[0].tolist()
        contexts = self.retriever.retrieve(query_vec, rewritten)
        return self.generator.generate(rewritten, contexts)


def get_rag_pipeline(cfg: RAGConfig | str) -> RAGPipeline:
    """Convenience wrapper to create a pipeline from ``RAGConfig`` or path."""
    if isinstance(cfg, (str, bytes, os.PathLike)):
        cfg = RAGConfig.from_yaml(cfg)
    return RAGPipeline(cfg)
