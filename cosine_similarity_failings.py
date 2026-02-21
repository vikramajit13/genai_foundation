from utils import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from common.constant import sentences



model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences, normalize_embeddings=False)

ignore_magnitude = cosine_similarity(embeddings[0], embeddings[1])
common_direction_bias = cosine_similarity(embeddings[2], embeddings[3])
false_positives = cosine_similarity(embeddings[4], embeddings[5])
no_notion_structure = cosine_similarity(embeddings[6], embeddings[7])

print("ignore_magnitude is ",ignore_magnitude)
print("common_direction_bias is ",common_direction_bias)
print("false_positives is ",false_positives)
print("no_notion_structure is ",no_notion_structure)
