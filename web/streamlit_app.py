%%writefile app.py
import torch 
import datasets 
from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain.docstore.document import Document as LangchainDocument
from datasets import load_dataset
from tqdm import tqdm
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig    

READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500,
)

def data_ingestion(dataset_name):
    dataset = load_dataset(dataset_name)
    RAW_KNOWLEDGE_BASE = [
    LangchainDocument(
        page_content=f"{ex['question']}\n\n{ex['answer']}", 
        metadata={"source": "stock_trading_QA"}
    ) 
    for ex in tqdm(dataset["train"]) 
    ]
    return RAW_KNOWLEDGE_BASE
    
def chunk_and_split_documents(chunk_size: int, knowledge_base: List[LangchainDocument], tokenizer_name) -> List[LangchainDocument]:
    # Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )
    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])
    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)
    return docs_processed_unique

def knowledge_vector_database(docs_processed, EMBEDDING_MODEL_NAME):
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    
    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    return KNOWLEDGE_VECTOR_DATABASE, embedding_model, tokenizer
#  common steps
# dataset_name = "yymYYM/stock_trading_QA"
# RAW_KNOWLEDGE_BASE = data_ingestion(dataset_name)
# EMBEDDING_MODEL_NAME = "thenlper/gte-small"
# docs_processed = chunk_and_split_documents(512, RAW_KNOWLEDGE_BASE, tokenizer_name=EMBEDDING_MODEL_NAME)
# KNOWLEDGE_VECTOR_DATABASE, embedding_model, tokenizer = knowledge_vector_database(docs_processed, EMBEDDING_MODEL_NAME)

def Retrieval(question, embedding_model, docs_processed, KNOWLEDGE_VECTOR_DATABASE):
    user_query = question
    query_vector = embedding_model.embed_query(user_query)
    embeddings_2d = [
        list(KNOWLEDGE_VECTOR_DATABASE.index.reconstruct_n(idx, 1)[0]) for idx in range(len(docs_processed))
    ] + [query_vector]
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
    return retrieved_docs

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

def main():
    st.title("RAG LLM with Streamlit")
    dataset_name = "yymYYM/stock_trading_QA"
    RAW_KNOWLEDGE_BASE = data_ingestion(dataset_name)
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    docs_processed = chunk_and_split_documents(512, RAW_KNOWLEDGE_BASE, tokenizer_name=EMBEDDING_MODEL_NAME)
    KNOWLEDGE_VECTOR_DATABASE, embedding_model, tokenizer = knowledge_vector_database(docs_processed, EMBEDDING_MODEL_NAME)
    question = st.text_input("Enter your question:")
    retrieved_docs = Retrieval(question, embedding_model, docs_processed, KNOWLEDGE_VECTOR_DATABASE)
    READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
    answer = Augmentation_and_Generation(READER_MODEL_NAME, tokenizer, retrieved_docs, question)
    st.write("**Answer:**", answer)
    # else:
    #     st.write("Please enter a question.")
if __name__ == "__main__":
    main()
