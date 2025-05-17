from importlib import metadata as _metadata
__version__ = _metadata.version(__name__.split('.')[0])

from .config import RAGConfig
from .pipelines import get_rag_pipeline
