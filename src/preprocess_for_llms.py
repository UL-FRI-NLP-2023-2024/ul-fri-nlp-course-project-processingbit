import tempfile
import requests
import torch
from transformers import AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import xml.etree.ElementTree as ET
import re
import pandas as pd
import numpy as np
from datasets import DatasetDict, load_dataset, Dataset
from tqdm import tqdm

#from IPython.display import print
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.docstore.document import Document

####### MODEL CONFIGURATION #######
with open("./tokens/hugging_face_token.txt", "r") as file:
    ACCESS_TOKEN = file.read().strip()

models = {
    "gemma": "google/gemma-7b", # NOT WORKING
    "mistral-22B": "mistralai/Mixtral-8x22B-Instruct-v0.1", # 
    "mistral-7B" : "mistralai/Mistral-7B-Instruct-v0.2",
    "llama-grand-2" : "meta-llama/Meta-Llama-Grand-2-8B",
    "llama-2-13B" : "meta-llama/Llama-2-13b-chat-hf", 
    "llama-2-70B" : "meta-llama/Llama-2-70b-chat-hf", 
    "llama-3-8" : "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-3-70" : "meta-llama/Meta-Llama-3-70B-Instruct",
}

LLM_MODEL = models["mistral-7B"]
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

####### FILE CONFIGURATION #######
CODEBOOK_FILE = "./data/codebook.xlsx"
DATASET_FILE = "./data/cleaned_data.csv"

######### PROMPT CONFIGURATION ########

INITIAL_PROMPT = """You are an AI expert in categorizing sentences into classes."""

CONTEXT = "Another assistant has retrieved some documents that might be useful for you to understand the context of the conversation, do not use them if not relevant."

CODEBOOK_PROMPT = """You can use the following codebook (with classes, definitions and examples) to help you ccategorize the sentence:
### IMPORTANT CODEBOOK:
{codebook}
###
You need to categorize the new sentence into one of the following classes: [{classes}].
If you fail to categorize the sentence, return 'None' instead of coming up with a wrong class.
"""

HISTORY = "The following is the history of the conversation:"


##### CODEBOOK FUNCTIONS  ######
def get_codebook():
    codebook = pd.read_excel(CODEBOOK_FILE)
    codebook[['Class', 'Term']] = codebook['Term'].str.split(":", expand=True)
    codebook['Term'] = codebook['Term'].map(lambda x: str(x.strip()))
    return codebook

def get_formatted_row(row):
    ### Add class name
    message_row = f"### Class: {row['Term']} ###\n"

    ### Add definition
    if not pd.isna(row['Definition']):
        definition = " ".join(row['Definition'].split('\n'))
        message_row += f"### Definition: {definition} ###\n"

    ### Add example
    if not pd.isna(row['Example']):
        example = " ".join(row['Example'].split('\n'))
        message_row += f"### Example: {example} ###\n"

    return message_row

def get_formatted_codebook(codebook, class_to_predict):
    codebook = codebook[codebook['Class'] == class_to_predict].copy()
    codebook.drop(columns=['Class'], inplace=True)
    formatted_codebook = codebook.apply(get_formatted_row, axis=1)
    formatted_codebook = "\n".join(formatted_codebook)
    return formatted_codebook

def get_classes_to_predict(codebook, class_to_predict):
    return codebook[codebook['Class'] == class_to_predict]['Term'].to_list()

def get_classes(codebook):
    return codebook['Class'].unique()


###### DATA PROCESSING FUNCTIONS ######
def preprocess_data(
        combine_fields = [],
        separator = ': ',
        text_field = 'message',
        class_field = '',
        history_field = 'past_chat',
        history_label = 'past_labels',
        unique_keys_for_conversation =  ['book_id', 'bookclub', 'course'],
        window_size = 3,
        use_past_labels = False
):
    data = pd.read_csv(DATASET_FILE)
    if len(combine_fields) > 0:
        data[text_field] = data[combine_fields].apply(lambda x: separator.join(x.dropna().astype(str)), axis=1)

    past_chat = []
    data[history_field] = pd.Series()
    if use_past_labels:
        past_labels = []
        data[history_label] = pd.Series()

    for i in range(len(data)):
        if i >= 1 and not data.iloc[i][unique_keys_for_conversation].equals(data.iloc[i-1][unique_keys_for_conversation]):
            past_chat = []
            if use_past_labels:
                past_labels = []

        data.at[i, history_field] = past_chat
        past_chat.append(data.iloc[i][text_field])
        
        if use_past_labels:
            data.at[i, history_label] = past_labels
            past_labels.append(data.iloc[i][class_field])

        if window_size > 0 and len(past_chat) > window_size:
            past_chat.pop(0)
            if use_past_labels:
                past_labels.pop(0)

    return data

def get_extension(class_to_predict, use_history, use_past_labels, use_context):
    extension = f'_{class_to_predict.lower()}'
    if use_history or use_past_labels or use_context:
        extension += '_w'
        extension += '_history' if use_history else ''
        extension += '_past-labels' if use_past_labels else ''
        extension += '_context' if use_context else ''
    return extension

##### MAIN FUNCTION ######

if __name__ == '__main__':

    use_history = True
    use_past_labels = True
    num_docs_as_context = 0

    class_to_predict = 'Discussion'

    model2id = {
        "mistral" : 0,
        "llama" : 1
    }
    model_used = model2id["mistral"]

    # get codebook
    codebook = get_codebook()
    classes = get_classes(codebook)
    
    print(f'Processing class: {class_to_predict}')
    # get formatted codebook and classes to predict
    formatted_codebook = get_formatted_codebook(codebook, class_to_predict)
    classes_to_predict = get_classes_to_predict(codebook, class_to_predict)

    # messages
    first_message = INITIAL_PROMPT
    codebook_message = CODEBOOK_PROMPT.format(codebook = formatted_codebook, classes = ", ".join(classes_to_predict))
        
    # Data processing
    text_field = 'message'
    history_field = 'past_chat'
    history_labels = 'past_labels'

    data = preprocess_data(
        combine_fields = [],
        separator = ': ',
        text_field = text_field,
        class_field= class_to_predict,
        history_field = history_field,
        history_label = history_labels,
        unique_keys_for_conversation =  ['book_id', 'bookclub', 'course'],
        window_size = 6,
        use_past_labels = use_past_labels
    )

    # getting the context
    if num_docs_as_context > 0:
        context = pd.read_csv("./preprocessed/context.csv")

    # Add input
    prompts = []
    for ind, row in data.iterrows():
        system_message = first_message

        if num_docs_as_context > 0:
            system_message += f"\n{CONTEXT}"
            docs = context.iloc[ind].to_list()
            for doc in docs:
                system_message += f"\n{doc}"

        system_message += f"\n{codebook_message}"

        if use_history:
            system_message += f"\n{HISTORY}"

        # No system for mistral
        if model_used == model2id["mistral"]:
            message = [{ "role": "user", "content": system_message}]
            message.append({ "role": "assistant", "content": "Ok, let's start!"})

        # System for llama
        if model_used == model2id["llama"]:
            message = [{ "role": "system", "content": system_message}]
        
        if use_history:
            for i, chat in enumerate(row[history_field]):
                message.append({ "role": "user", "content": chat})
                if use_past_labels:
                    message.append({ "role": "assistant", "content": f'Class: {row[history_labels][i]}'})
        
        message.append({ "role": "user", "content": row[text_field]})
        prompts.append(message)

    # COMBINE PROMPTS WITH FINAL CLASSES IN PANDAS
    labels = [str(label) for label in data[class_to_predict]]
    indexes = data.index

    final_dataset = Dataset.from_dict({
        'index': indexes,
        'text': prompts,
        'labels': labels
    })

    # Split into train and test
    final_dataset = final_dataset.train_test_split(test_size=0.2, seed=42)
    print('Final dataset:', final_dataset)

    # Save final data
    final_path = f"./preprocessed/data_{get_extension(class_to_predict, use_history, use_past_labels, num_docs_as_context > 0)}"

    final_dataset.save_to_disk(final_path)
