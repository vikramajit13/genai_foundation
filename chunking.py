from typing import List

def chunk_text(text: str, *, chunk_size: int = 150, overlap: int = 100) -> List[str]:
    """
    Splits text into overlapping chunks of a fixed character length.
    """
    start: int = 0
    chunks: List[str] = [] # type: ignore

    while start < len(text):
        end: int = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)

    return chunks

str = "For a Senior Node.js role handling 30,000 sensors and satellite communications, the interview will move beyond basic syntax to focus on high-throughput system design, latency management, and resource efficiency."
print(len(chunk_text(str)))
