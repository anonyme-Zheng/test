from __future__ import annotations
import logging, json, pickle, gzip, pathlib
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

def save_json(obj, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path,'r',encoding='utf-8') as f:
        return json.load(f)
