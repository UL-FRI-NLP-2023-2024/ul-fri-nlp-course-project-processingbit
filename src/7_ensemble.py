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

from utils import *


ACCESS_TOKEN = get_access_token()

model_name = "mistral-8B"
LLM_MODEL = get_model_path(model_name)
print(f'Model: {LLM_MODEL}')

PATH_DIR = "./pred_results/"



def parse_filename(filename):
    """
    Parse the filename to extract the method used and the features encoded in the extension.
    """
    # Remove the .npy extension to ease parsing
    base_name = filename[:-4]
    
    # Split the base_name by '_' to separate method_used from the extension
    parts = base_name.split('_')
    method_used = parts[0]
    features = parts[1:]
    
    return method_used, features

def match_features(features):
    """
    Match the features list with the corresponding boolean flags.
    """
    class_to_predict = features[0]  # The first part after method_used is always class_to_predict
    use_history = 'history' in features
    use_past_labels = 'past-labels' in features
    use_context = 'context' in features
    
    return class_to_predict, use_history, use_past_labels, use_context


def load_data_predicts(class_to_predict):
    dataset = load_from_disk(f"./preprocessed/data_{get_extension(class_to_predict, False, False, False)}")
    data = pd.DataFrame(dataset['test'])

    # Loop through each .npy file in the directory
    for file in os.listdir(PATH_DIR):
        if file.endswith(".npy"):
            method_used, features = parse_filename(file)
            class_to_predict, use_history, use_past_labels, use_context = match_features(features)
            if class_to_predict == class_to_predict:
                name = file[:-4]
                predictions = np.load(PATH_DIR + file)
                data[name] = predictions
    
    return data

