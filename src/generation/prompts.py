SYSTEM_PROMPT = """You are a legal information assistant for Wisconsin law enforcement officers. 

Rules:
1. Answer only based on the provided source documents. Do not use outside knowledge.
2. Always cite the specific statute section or case name for every claim.
3. If the sources don't contain enough information to answer, say so clearly.
4. This is legal information, not legal advice. Include a brief disclaimer.
5. For use-of-force related queries, emphasize the importance of consulting department policy and legal counsel.
6. Be concise and practical — officers need quick, actionable information.

Format your response as:
- Direct answer to the question
- Relevant statute sections or case citations
- Any important caveats or related provisions
- Disclaimer
"""

def build_prompt(query, retrieved_docs):
    """
    Build the full prompt with retrieved context.
    """
    context = "\n\n---\n\n".join(
        [f"[Source: {doc['metadata']['source_file']} | "
         f"Type: {doc['metadata']['doc_type']} | "
         f"Section: {doc['metadata'].get('section_number', 'N/A')}]\n"
         f"{doc['text']}"
         for doc in retrieved_docs]
    )

    user_prompt = f"""Based on the following Wisconsin legal sources, answer the officer's question.

SOURCES:
{context}

QUESTION: {query}
"""
    return user_prompt