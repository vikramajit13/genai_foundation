sentence_1 = "This is a test sentence for embeddings"
sentence_2 = "this is also a test sentence for embeddings"
sentence_3= "this could have been a test sentence for embedding but has been changes slightly"
sentence_4 = "The weather today is hot and humid"

import numpy as np
from sentence_transformers import SentenceTransformer

sentences= [
    "This is a test sentence for embeddings",
    "this is also a test sentence for embeddings",
    "this could have been a test sentence for embedding but has been changes slightly",
    "The weather today is hot and humid"
]

def cosine_similarity(a,b) -> float:
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings =  model.encode(sentences, normalize_embeddings=False)

n= len(sentences)
print("Cosine similarity matrix:\n")

for i in range(n):
    row =[]
    for j in range(n):
        sim = cosine_similarity(embeddings[i],embeddings[j])
        row.append(f"{sim:0.3f}")
        print(" ".join(row))

print("respone from cosine_similarity is :::")

#print(cosine_similarity(sentence_1, sentence_2))