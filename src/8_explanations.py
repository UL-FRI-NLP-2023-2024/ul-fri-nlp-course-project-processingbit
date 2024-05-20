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




def build_message(data, class_to_predict):
    
    all_message = data['text']
    
    new_initial_prompt = "You are an ensemble model. You have to predict the class of the following text, based on the results of other models."


    return message

if __name__ == "__main__":
    class_to_predict = "Discussion"
    path_dir = "./pred_results/"

    model_name = "llama-3-8"
    llm_model = get_model_path(model_name)
    print(f'Model: {llm_model}')

    data = load_data_predicts(path_dir, class_to_predict, model_name)
    
    message = build_message(data, class_to_predict)

    # Load the model
    #model = get_model(model_name, quantize=True)
    #tokenizer = get_tokenizer(model_name)
