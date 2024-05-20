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


if __name__ == "__main__":
    class_to_predict = "Discussion"
    path_dir = "./pred_results/"

    model_name = "llama-3-8"
    llm_model = get_model_path(model_name)
    print(f'Model: {llm_model}')

    data = load_data_predicts(path_dir, class_to_predict, model_name)
    
    # Message used to ask the llm for an explanation
    explanation_message = "Based on the previous chat, can you explain to me why you have chosen the last class?"

    # Retrieve one message from the test data for proof of concept
    test_data = data.iloc[0]
    test_index = test_data['index']
    test_message = test_data['text']
    test_label = test_data['labels']
    label_from_model = "Seminar"

    test_message.append({
        "role": "assistant",
        "content": f"Class: {label_from_model}"
        }
    )
    test_message.append({
        "role": "user",
        "content": explanation_message
        }
    )

    # Load the model
    model = get_model(model_name, quantize=True)
    tokenizer = get_tokenizer(model_name)

    # Format the text
    test_message = tokenizer.apply_chat_template(test_message, tokenize=False, add_generation_prompt=True)
    print(test_message)

    # Get the explanation
    explanation = model.generate_text(test_message, max_length=512)
    print(explanation)
