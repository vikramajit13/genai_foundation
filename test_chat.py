import ollama
from typing import List

def query_with_context(user_query: str, chunks: List[str], selected_indices: List[int]) -> str:
    """
    Invokes Ollama with a structured system prompt and context chunks.
    """
    # 1. Prepare the context string from the selected indices
    # This joins your top 3 chunks into the required format
    context_text = "\n".join([f"<{chunks[i]}>" for i in selected_indices])

    # 2. Construct the prompt following your specific format
    full_prompt = f"""
You are a helpful assistant.
Use ONLY the context below to answer.

Context:
{context_text}

Question:
{user_query}
"""

    # 3. Invoke the model
    response = ollama.chat(
        model='llama3.1:8b', # or your preferred model
        messages=[
            {'role': 'user', 'content': full_prompt},
        ],
    )

    return response.message.content

# Example Usage:
# top_indices = [2, 5, 0] 
# result = query_with_context("What is the main topic?", my_chunks, top_indices)
# print(result)

