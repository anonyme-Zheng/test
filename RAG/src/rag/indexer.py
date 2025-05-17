from elasticsearch import Elasticsearch, helpers
import numpy as np
from .config import ESConfig
from .utils.io import logger

class ESIndexer:
    def __init__(self, cfg:ESConfig):
        self.cfg = cfg
        self.es = Elasticsearch(
            hosts=[cfg.host],
            basic_auth=(cfg.username, cfg.password) if cfg.username else None,
            request_timeout=30, max_retries=10, retry_on_timeout=True)
        self._ensure_index()

    def _ensure_index(self):
        if self.es.indices.exists(index=self.cfg.index_name): return
        body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "vector": {"type": "dense_vector", "dims": self.cfg.embedding_dim, "index": True, "similarity": "cosine"},
                    "source_id": {"type": "keyword"},
                }
            }
        }
        self.es.indices.create(index=self.cfg.index_name, body=body)
        logger.info("Created ES index %s", self.cfg.index_name)

    def index_chunks(self, chunks:list[dict], vectors:np.ndarray):
        assert len(chunks)==len(vectors)
        actions = (
            {
                "_index": self.cfg.index_name,
                "_id": ch["chunk_id"],
                "_source": {**ch, "vector": vec.tolist()}
            } for ch,vec in zip(chunks,vectors)
        )
        helpers.bulk(self.es, actions)
        logger.info("Indexed %d chunks", len(chunks))
