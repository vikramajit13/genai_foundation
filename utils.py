import numpy as np
import re
from typing import List

def cosine_similarity(a, b) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def split_sentences(text: str):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def chunk_text(
    text: str, target_chars: int = 800, overlap_sents: int = 1, min_chars: int = 200
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


def keyword_score(query: str, chunk: str) -> float:
    q_words = set(query.lower().split())
    c_words = set(chunk.lower().split())
    overlap = q_words.intersection(c_words)
    return len(overlap) / (len(q_words) + 1e-6)


STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","is","are","was","were",
    "be","been","being","that","this","it","as","at","by","from","but","not","all",
    "they","their","them","we","you","your","i","our","us"
}

def tokenize(text: str) -> list[str]:
    # keep numbers and % as tokens, drop most punctuation
    tokens = re.findall(r"[a-zA-Z]+|\d+%?|\d+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]



import re
from typing import List

def chunk_by_sections(text: str) -> List[str]:
    text = text.strip()

    # Split on roman numeral headings like "I.", "II.", ... at start of line
    # Keep the heading with the section.
    parts = re.split(r"\n(?=(I|II|III|IV|V)\.\s)", text)

    # re.split with capturing group gives: [preamble, "I", rest, "II", rest, ...]
    # We’ll stitch them back.
    chunks = []
    if len(parts) == 1:
        return [text]

    preamble = parts[0].strip()
    if preamble:
        chunks.append(preamble)

    i = 1
    while i < len(parts) - 1:
        numeral = parts[i].strip()
        body = parts[i+1].strip()
        chunks.append(f"{numeral}. {body}".strip())
        i += 2

    # Final cleanup: drop tiny chunks
    chunks = [c for c in chunks if len(c) > 200]
    return chunks