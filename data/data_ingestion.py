dataset_name = "yymYYM/stock_trading_QA"
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
