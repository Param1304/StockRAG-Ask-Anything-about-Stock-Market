def Augmentation_and_Generation(model_name, tokenizer, retrieved_docs, question):
    prompt_in_chat_format = [
    {
        "role": "system",
        "content": """ You are a Stock market expert. Using the information contained in the context,
        give a comprehensive answer to the question.
        Respond only to the question asked, response should be concise and relevant to the question.
        Provide the number of the source document when relevant.
        If the answer cannot be deduced from the context, do not give an answer.""",
    },
    {
        "role": "user",
        "content": """Context:
        {context}
        ---
        Now here is the question you need to answer.

        Question: {question}""",
    },
]
    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]  # We only need the text of the documents
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])
    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)
    answer = READER_LLM(final_prompt)[0]["generated_text"]
    return answer
