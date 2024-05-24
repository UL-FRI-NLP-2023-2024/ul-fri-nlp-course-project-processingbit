import pandas as pd
from utils import *

"""
This script performs data processing and retrieval using the RAG (Retrieval-Augmented Generation) model.

The script reads a dataset file containing cleaned data and preprocesses it for input to the RAG model. 
It then uses a retriever to retrieve relevant documents based on the input prompts. 
The retrieved documents are saved to a CSV file.

Configuration Options:
    - use_history: Whether to include chat history in the input prompts.
    - quantize: Whether to quantize the retrieved documents.
    - window_size: The size of the window for considering past labels.
    - dataset_file: The path to the dataset file containing cleaned data.
    - retriever: The retriever object used for document retrieval.
    - text_field: The name of the field in the dataset containing the input text.
    - history_field: The name of the field in the dataset containing the chat history.

Output:
    - The retrieved documents are saved to a CSV file named 'context.csv' in the 'preprocessed' directory.
"""

if __name__ == '__main__':

    use_history = True
    quantize = True
    window_size = -1
    dataset_file = "./data/cleaned_data.csv"

    # Use default llm and retirever
    retriever = Retriever(xmls = ['./data/LadyOrThetigerIMapBook.xml'], 
                            documents_to_retrieve = 5,
                            use_llm=True,
                            quantize = quantize,
                            )
    
    # Data processing
    text_field = 'message'
    history_field = 'history'

    data = preprocess_data(
        dataset_file=dataset_file,
        text_field = text_field,
        history_field = history_field,
        window_size = window_size,
        use_past_labels=False,
    )

    # Prompt requests with input and w/wo history
    if use_history:
        prompt_requests = data.apply(lambda x: {'input': x[text_field],
                                                'chat_history': x[history_field],
                                                }, axis=1).to_list()
    else:
        prompt_requests = data.apply(lambda x: {'input': x[text_field]}, axis=1).to_list()

    
    # Retrieval
    docs_data = retriever.batch_invoke(prompt_requests, history_field = 'chat_history')

    dataset = pd.DataFrame(docs_data)

    # Save dataset
    dataset.to_csv(f"./preprocessed/context.csv", index=False)
