import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from prompts import SYSTEM_PROMPT, build_prompt

load_dotenv()


def get_llm_response(query, retrieved_results):
    """
    Send query + retrieved context to OpenAI and get a response.
    """
    
    # Format retrieved results into docs with text and metadata
    docs = []
    for i in range(len(retrieved_results["documents"][0])):
        docs.append({
            "text": retrieved_results["documents"][0][i],
            "metadata": retrieved_results["metadatas"][0][i],
        })

    user_prompt = build_prompt(query, docs)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # Low temperature for factual accuracy
    )

    # Deduplicate sources by (source_file, section_number) — keep first occurrence
    seen = set()
    unique_sources = []
    for doc in docs:
        meta = doc["metadata"]
        key = (meta.get("source_file", ""), meta.get("section_number", ""))
        if key not in seen:
            seen.add(key)
            unique_sources.append(meta)

    return {
        "answer": response.choices[0].message.content,
        "sources": unique_sources,
    }


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from retrieval.vector_store import query_vector_store

    query = "Can I search a vehicle during a traffic stop without consent?"
    print(f"Query: {query}\n")

    results = query_vector_store(query, n_results=5)
    response = get_llm_response(query, results)

    print("ANSWER:")
    print(response["answer"])
    print("\nSOURCES:")
    for s in response["sources"]:
        print(f"  - {s}")