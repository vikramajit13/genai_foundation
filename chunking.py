def chunk_text(text, *, chunk_size=150, overlap=100):
    start = 0
    chunks = []

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


str = "For a Senior Node.js role handling 30,000 sensors and satellite communications, the interview will move beyond basic syntax to focus on high-throughput system design, latency management, and resource efficiency."
print(len(chunk_text(str)))
