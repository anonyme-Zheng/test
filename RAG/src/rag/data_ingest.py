"""Clean, segment and yield chunks from raw annual-report documents."""
from __future__ import annotations
from pathlib import Path
import re, itertools
from typing import Iterable, Dict

_CHUNK_WORDS = 180  # roughly 512 tokens

def _clean(text:str)->str:
    text = re.sub(r"\\s+", " ", text)
    return text.strip()

def segment(text:str, source_id:str, chunk_words:int=_CHUNK_WORDS)->Iterable[Dict]:
    words = text.split()
    for i in range(0, len(words), chunk_words):
        chunk = " ".join(words[i:i+chunk_words])
        yield {
            "source_id": source_id,
            "chunk_id": f"{source_id}_{i//chunk_words}",
            "text": chunk,
        }

def iterate_raw_files(data_dir: str|Path) -> Iterable[Dict]:
    data_dir = Path(data_dir)
    for p in data_dir.glob("*.txt"):
        text = _clean(p.read_text(encoding='utf-8'))
        yield from segment(text, p.stem)