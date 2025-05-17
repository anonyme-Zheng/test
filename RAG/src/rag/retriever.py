import numpy as np
from typing import List
from .config import ESConfig
from elasticsearch import Elasticsearch
from .utils.io import logger

class HybridRetriever:
    def __init__(self, cfg:ESConfig, top_k:int=10):
        self.es = Elasticsearch(hosts=[cfg.host])
        self.index = cfg.index_name
        self.top_k = top_k

    def retrieve(self, query_vec:list[float], query_txt:str)->List[dict]:
        body = {
            "size": self.top_k,
            "query": {
                "bool": {
                    "should": [
                        {"knn": {"vector": {"vector": query_vec, "k": self.top_k}}},
                        {"match": {"text": query_txt}}
                    ]
                }
            }
        }
        res = self.es.search(index=self.index, body=body)
        hits = [h["_source"] | {"score": h["_score"]} for h in res["hits"]["hits"]]
        logger.debug("Retrieved %d hits", len(hits))
        return hits