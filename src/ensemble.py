import os
import tempfile
from pathlib import Path
import joblib
import requests
import torch
from transformers import AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline, TrainingArguments
import xml.etree.ElementTree as ET
import re
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from accelerate import PartialState
import tqdm

# Spelling
from spellchecker import SpellChecker
from nltk.stem import SnowballStemmer, PorterStemmer
import nltk
from nltk.corpus import stopwords 

#from IPython.display import print
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.docstore.document import Document

from peft import LoraConfig, PeftConfig, get_peft_model, prepare_model_for_kbit_training

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

with open("./tokens/hugging_face_token.txt", "r") as file:
    ACCESS_TOKEN = file.read().strip()
    
models = {
    "mistral-22B": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mistral-7B" : "mistralai/Mistral-7B-Instruct-v0.2",
    "llama-grand-2" : "meta-llama/Meta-Llama-Grand-2-8B",
    "llama-2-13B" : "meta-llama/Llama-2-13b-chat-hf", 
    "llama-2-70B" : "meta-llama/Llama-2-70b-chat-hf", 
    "llama-3-8" : "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-3-70" : "meta-llama/Meta-Llama-3-70B-Instruct",
}

LLM_MODEL = models["mistral-7B"]
print(f'Model: {LLM_MODEL}')

PATH_DIR = "./results/"

stemmer = PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
spell = SpellChecker()

def preprocess(text):
    tokens = []
    for token in nltk.word_tokenize(text.lower()):
        if token.isalpha() and token not in stop_words:
            token = spell.correction(token)
            if token is not None:
                token = lemmatizer.lemmatize(token)
                token = stemmer.stem(token)
                tokens.append(token)

    return " ".join(tokens)

def find_first(sentence, items):
    sentence = preprocess(sentence)
    first = len(sentence)
    item = None
    for i in items:
        index = sentence.find(preprocess(i))
        if index != -1 and index < first:
            first = index
            item = i
    if item is None:
        item = 'None'
    return item

def get_extension(class_to_predict, use_history, use_context):
    extension = f'_{class_to_predict}'
    extension += '_with_history' if use_history else ''
    extension += '_with_context' if use_context else ''
    return extension

def load_data(class_to_predict, use_history, use_context):
    dataset = load_from_disk(f"./preprocessed/dataset_{get_extension(class_to_predict, use_history, use_context)}")
    data = pd.DataFrame(dataset['test'])


    for file in os.listdir(PATH_DIR):
        
        
            

    return data