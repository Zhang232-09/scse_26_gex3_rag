### COMPLETE THE CODE ###
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def retrieve(query: str, documents, top_k: int = 2):
    """
    Perform similarity retrieval from langchain Document list.
    Args:
        query: user question string
        documents: List[Document] from load_policy_documents() (single‑file knowledge base)
        top_k: return top N most relevant text chunks
    Returns:
        list[str]: page_content of relevant documents
    """
    if not documents:
        return []

    corpus = [doc.page_content for doc in documents]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus + [query])

    doc_vectors = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1]

    similarity_scores = cosine_similarity(query_vector, doc_vectors)[0]
    sorted_indices = similarity_scores.argsort()[::-1][:top_k]

    retrieved_texts = [corpus[i] for i in sorted_indices]
    return retrieved_texts
