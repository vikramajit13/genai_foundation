from sentence_transformers import SentenceTransformer
from utils import cosine_similarity
from chunking import chunk_text

# sentences= [
#     "This is a test sentence for embeddings",
#     "this is also a test sentence for embeddings",
#     "this could have been a test sentence for embedding but has been changes slightly",
#     "The weather today is hot and humid"
# ]

str_text = """Backpressure & Streams: How would you handle a situation where sensors are sending data faster than your database can write? Expect questions on using Node.js Streams and handling backpressure to prevent memory overflows.
* Worker Threads vs. Clustering: You may be asked how to utilize multi-core systems to process sensor payloads without blocking the main thread. Be ready to differentiate between the Cluster Module (for scaling across cores) and Worker Threads (for CPU-intensive data parsing).
* Data Serialization: Questions on using Protocol Buffers (Protobuf) or MessagePack instead of JSON to reduce payload size—crucial for satellite links with limited bandwidth. 
"""

sentences = chunk_text(str_text)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences, normalize_embeddings=False)

n = len(sentences)
print("Cosine similarity matrix:\n",n)

for i in range(n):
    row = []
    for j in range(n):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        row.append(f"{sim:0.3f}")
    print(" ".join(row))

print("respone from cosine_similarity is :::")

# print(cosine_similarity(sentence_1, sentence_2))
