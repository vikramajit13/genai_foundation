
from common.embedings import return_embeddings
from common.invoke_llama_chat import query_with_context
from utils import cosine_similarity, chunk_text, chunk_by_sections
import numpy as np
from pprint import pprint
from retrievalidf.tfidf import build_idf, keyword_score_idf


str = """
SUBJECT: 2024 State of the Union Address - Key Pillars

I. Freedom and Democracy Under Attack
President Biden began by referencing the 1941, comparing the current era to a time when democracy was under assault, both domestically and abroad [1]. He emphasized that Russia's invasion of Ukraine threatens global stability and asserted that Ukraine could defend itself with continued U.S. support [1]. The address criticized those blocking assistance for Ukraine and emphasized the need for continued global leadership [1]. 
II. The Great American Comeback
The President described an economic recovery, characterizing it as the "greatest comeback story never told" [1]. He highlighted the "middle-out" economic approach, aiming to invest in all Americans [1]. Key accomplishments mentioned included 15 million new jobs in three years, low unemployment, and a record 16 million new business applications [1]. He also noted significant job growth, including in the manufacturing sector [1]. 
III. Reproductive Freedom and Fundamental Rights
Biden addressed the issue of reproductive rights, specifically highlighting the Alabama Supreme Court's ruling on IVF treatments, which he termed a result of overturning Roe v. Wade [1]. He called for Congress to pass legislation to protect IVF nationwide [1]. Furthermore, he thanked Vice President Harris for her work on reproductive freedom and emphasized the political power of women in influencing elections [1]. 
IV. Tax Fairness and Big Pharma
The President criticized the high cost of prescription drugs, citing the law that allows Medicare to negotiate prices and caps insulin costs at $35 a month for seniors, which he intends to expand to all Americans [1]. He advocated for a more equitable tax system, arguing that the wealthy and large corporations should pay a fair share [1]. He criticized the previous administration's $2 trillion tax cut and vowed that no one earning less than $400,000 would pay higher federal taxes [1]. 
V. National Security and The Middle East
Addressing the conflict in the Middle East, Biden described the October 7th Hamas attack as a tragic event [1]. He supported Israel's right to defend itself against Hamas, but stressed the necessity for protecting innocent civilians in Gaza [1]. He announced a U.S. military mission to establish a temporary pier in the Mediterranean to deliver humanitarian aid, while emphasizing the need for Israel to allow more assistance into the region [1]
"""

#query = "Based on the 2024 State of the Union Address, what are the primary challenges or actions being addressed across the areas of democracy, corporate taxation, and reproductive rights?"

query = (
  "From the context, answer ALL of the following with a direct quote for each:\n"
  "1) democracy/foreign policy challenge,\n"
  "2) the corporate minimum tax change (include the percentages),\n"
  "3) one reproductive rights point.\n"
  "If any item is missing, say 'Not found in provided context' for that item."
)



sentences = chunk_text(str)
pprint(sentences)

idf = build_idf(sentences)
# keyword_score = keyword_score_idf(query,sentences[0], idf)
# print("keyword_score")
# print(keyword_score)

embeddings_document = return_embeddings(sentences)
embeddings_query = return_embeddings([query])[0]

keyword_boosts = [1.0 if "tax" in c.lower() else 0.0 for c in sentences]

## find the shortest chunks in the list and penalise them
lengths_list = list(map(len, sentences))

n = len(sentences)
print("total_Embeddings::", n)
scores = []
for idx, chunk_emb in enumerate(embeddings_document):
    sim = float(cosine_similarity(embeddings_query, chunk_emb))
    scores.append((idx, sim))

final_rankings = []
for i in range(n):
    penalty = np.log10(lengths_list[i] + 1) / np.log10(np.max(lengths_list) + 1)
    final_scores = 0.75 * scores[i][1] + 0.2 * keyword_score_idf(query,sentences[i], idf) + 0.05*penalty
    #final_scores = scores[i][1] 
    final_rankings.append((scores[i][0], final_scores))


final_rankings.sort(key=lambda x: x[1], reverse=True)
final_results_synonym = [idx for idx, _ in final_rankings]
topk_syn_final = final_results_synonym[:3]

print("invoking llama with below top hcunk")
print(topk_syn_final[0])

print("\n=== RETRIEVAL TRACE ===")
for rank, (idx, score) in enumerate(final_rankings, start=0):
    print("score:::")
    print(score)
 
    preview = sentences[idx].replace("\n", " ")[:120]
   
    print(f"Rank {rank} | idx={idx} | score={score:.3f} | {preview}...")

# print("\n=== CONTEXT SENT TO LLM (first 800 chars) ===")
# print(context[:800])
# print("Context chars:", len(context))
response = query_with_context(query, sentences, topk_syn_final)

pprint(response)
