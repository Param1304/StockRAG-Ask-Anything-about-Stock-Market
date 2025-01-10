def Retrieval(question, embedding_model, docs_processed, KNOWLEDGE_VECTOR_DATABASE):
    user_query = question
    query_vector = embedding_model.embed_query(user_query)
    embeddings_2d = [
        list(KNOWLEDGE_VECTOR_DATABASE.index.reconstruct_n(idx, 1)[0]) for idx in range(len(docs_processed))
    ] + [query_vector]
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
    return retrieved_docs
