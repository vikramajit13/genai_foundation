from typing import List, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from chunking import chunk_text

def return_embeddings(sentences: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Chunks text and returns a NumPy array of embeddings.
    """
    #sentences: List[str] = chunk_text(str_text)
    
    # Initialize model
    model = SentenceTransformer(model_name)
    
    # model.encode returns a numpy.ndarray by default
    embeddings: np.ndarray = model.encode(sentences, normalize_embeddings=True)
    
    return embeddings