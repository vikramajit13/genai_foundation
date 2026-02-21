
from utils import cosine_similarity
from sentence_transformers import SentenceTransformer
from chunking import chunk_text
import numpy as np
from pprint import pprint
query = "How do I prevent memory overflow when sensors send data faster than DB writes?"
query_synonym_poor = "In the context of recurrent LLM inference and associative recall (AR) tasks, how can I implement a chunk-based inference strategy to mitigate memory overflow and prevent the effective state capacity from being exceeded during long-context decoding, specifically focusing on HBM (High-Bandwidth Memory) allocation for K,V cache prefetching and the prevention of heap-based buffer overflow vulnerabilities (CWE-122) through address space layout randomization (ASLR) and data execution prevention (DEP) to ensure semantic caching integrity?"

str_text = """Backpressure & Streams: How would you handle a situation where sensors are sending data faster than your database can write? Expect questions on using Node.js Streams and handling backpressure to prevent memory overflows.
* Worker Threads vs. Clustering: You may be asked how to utilize multi-core systems to process sensor payloads without blocking the main thread. Be ready to differentiate between the Cluster Module (for scaling across cores) and Worker Threads (for CPU-intensive data parsing).
* Data Serialization: Questions on using Protocol Buffers (Protobuf) or MessagePack instead of JSON to reduce payload size—crucial for satellite links with limited bandwidth. 
"""

sentences = chunk_text(str_text)
#pprint(sentences)
## find the shortest chunks in the list and penalise them 
lengths_list = list(map(len, sentences))
max = np.max(lengths_list) * 10
pprint(np.max(lengths_list))
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences, normalize_embeddings=True)
query_embed =model.encode([query],normalize_embeddings=True)[0]
query_embed_synonym =model.encode([query_synonym_poor],normalize_embeddings=True)[0]

keyword_boosts = [1.0 if "overflow" in c.lower() else 0.0 for c in sentences]


n = len(sentences)
print("total_Embeddings::",n)
scores=[]
score_synonym=[]
for idx, chunk_emb in enumerate(embeddings):
    sim=cosine_similarity(query_embed,chunk_emb)
    sim_synonym = cosine_similarity(query_embed_synonym,chunk_emb)
    scores.append((idx,sim))
    score_synonym.append((idx,sim_synonym))


## use Keyword boost to return values
final_rankings = []  
for i in range(n):
    penalty = np.log10(lengths_list[i] + 1) / np.log10(np.max(lengths_list) + 1)
    final_scores = 0.6 * scores[i][1] + 0.2 *keyword_boosts[i] + 0.2 * penalty
    final_rankings.append((scores[i][0], final_scores))
    
#pprint(final_rankings)

scores.sort(key=lambda x:x[1],reverse=True)
score_synonym.sort(key=lambda x:x[1],reverse=True)
final_rankings.sort(key=lambda x:x[1],reverse=True)
results = [idx for idx, _ in scores]   # ranked chunk indices
topk = results[:3]                      # show top 3 with scores
print("Top 3:", topk)
results_synonym = [idx for idx, _ in score_synonym]
topk_syn = results_synonym[:3]
print("Top 3:", topk_syn)

final_results_synonym = [idx for idx, _ in final_rankings]
topk_syn_final= final_results_synonym[:3]
print("final Top 3:", topk_syn_final)



correct_chunk_index = 0

def evaluate_recall_at_k(results, correct_chunk_index, k):
    return 1 if correct_chunk_index in results[:k] else 0

print("Recall@1:", evaluate_recall_at_k(results, correct_chunk_index, 1))
print("Recall@3:", evaluate_recall_at_k(results, correct_chunk_index, 3))