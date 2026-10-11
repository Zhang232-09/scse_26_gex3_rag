### COMPLETE THE CODE ###
from rag_tool import retrieve


def run_rag(user_query: str, documents):
    """
    Only perform retrieval, return relevant text chunks.
    No LLM generation here to avoid circular import.
    Args:
        user_query: input question string
        documents: Document list loaded by policy_loader
    Returns:
        list[str]: retrieved context chunks
    """
    retrieved_chunks = retrieve(user_query, documents)
    return retrieved_chunks 
