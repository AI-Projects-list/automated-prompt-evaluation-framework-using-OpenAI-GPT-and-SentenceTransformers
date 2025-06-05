import openai
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

openai.api_key = "sk-..."  # Replace with your actual OpenAI API key

df = pd.read_csv("prompts.csv")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_openai_response(prompt, model="gpt-3.5-turbo"):
    try:
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return completion.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error: {e}")
        return "Error"

results = []
for index, row in df.iterrows():
    prompt = row['prompt']
    expected = row['expected']
    response = get_openai_response(prompt)

    emb_expected = embed_model.encode([expected])[0]
    emb_response = embed_model.encode([response])[0]
    similarity = cosine_similarity([emb_expected], [emb_response])[0][0]

    results.append({
        "id": row["id"],
        "prompt": prompt,
        "expected": expected,
        "response": response,
        "similarity": round(similarity, 3)
    })

    print(f"[{row['id']}] Similarity: {round(similarity, 3)}")
    time.sleep(1)

result_df = pd.DataFrame(results)
result_df.to_csv("results.csv", index=False)
print("\n✅ Evaluation complete. Results saved to results.csv")
