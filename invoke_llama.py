from chunking import chunk_text
from common.embedings import return_embeddings
from common.invoke_llama_chat import query_with_context
from utils import cosine_similarity
import numpy as np
from pprint import pprint


str = """
SUBJECT: 2024 State of the Union Address - Key Pillars

PART 1: FOREIGN POLICY AND DEMOCRACY
Freedom and democracy are under attack both at home and overseas... Putin of Russia is on the march, invading Ukraine...

PART 2: THE AMERICAN ECONOMY
...building an economy from the middle out and the bottom up... raising the corporate minimum tax from 15% to at least 21%.

PART 3: HEALTHCARE AND REPRODUCTIVE RIGHTS
...defending reproductive freedom... The Supreme Court majority wrote, 'Women are not without electoral or political power.'
"""

query = "What is the proposed change to the corporate minimum tax rate?"



sentences = chunk_text(str, chunk_size=100, overlap=50)

embeddings_document = return_embeddings(sentences)
embeddings_query = return_embeddings([query])

keyword_boosts = [1.0 if "tax" in c.lower() else 0.0 for c in sentences]

## find the shortest chunks in the list and penalise them
lengths_list = list(map(len, sentences))

n = len(sentences)
print("total_Embeddings::", n)
scores = []
for idx, chunk_emb in enumerate(embeddings_document):
    sim = cosine_similarity(embeddings_query, chunk_emb)
    scores.append((idx, sim))

final_rankings = []
for i in range(n):
    penalty = np.log10(lengths_list[i] + 1) / np.log10(np.max(lengths_list) + 1)
    final_scores = 0.6 * scores[i][1] + 0.2 * keyword_boosts[i] + 0.2 * penalty
    final_rankings.append((scores[i][0], final_scores))


final_rankings.sort(key=lambda x: x[1], reverse=True)
final_results_synonym = [idx for idx, _ in final_rankings]
topk_syn_final = final_results_synonym[:3]

print("invoking llama with below top hcunk")
print(topk_syn_final[0])
response = query_with_context(query, sentences, topk_syn_final)

pprint(response)
