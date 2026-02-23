import numpy as np
import re
from typing import List


def cosine_similarity(a, b) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def split_sentences(text: str):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def chunk_text(
    text: str, target_chars: int = 1200, overlap_sents: int = 1, min_chars: int = 200
) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = []

    def flush():
        if not buf:
            return
        chunk = "\n\n".join(buf).strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)

    for p in paras:
        parts = [p]
        if len(p) > target_chars * 1.5:
            parts = split_sentences(p)
            
        for part in parts:
            if sum(len(x) for x in buf) + len(part) +2 <= target_chars:
                buf.append(part)
            else:
                flush()
                if overlap_sents > 0 and chunks:
                    # take last overlap_sents sentences from previous chunk
                    prev = chunks[-1]
                    tail = split_sentences(prev)[-overlap_sents:]
                    buf = [" ".join(tail), part]
                else:
                    buf = [part]
                
    flush()
    return chunks