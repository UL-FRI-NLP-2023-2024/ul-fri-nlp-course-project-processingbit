import os
import tempfile
from pathlib import Path
import joblib
import requests
import torch
from transformers import AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline, TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding
import xml.etree.ElementTree as ET
import re
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from accelerate import PartialState

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

from peft import LoraConfig, PeftConfig, get_peft_model

from trl import SFTTrainer 
import os

print(os.getcwd())

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

LLM_MODEL = models["llama-3-8"]
print(f'Model: {LLM_MODEL}')

class_to_predict = 'Discussion'
quantize = False

dataset_csv_file =  './data/cleaned_data.csv'
window_size = 3
text_field = 'prompt'

use_adapters = False

def split_data(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Training dataset
    labels = train_data[class_to_predict]
    prompts = train_data[text_field]

    train_prompts = DatasetDict({
        'train': Dataset.from_dict({'text': prompts, 'label': labels})
    })

    # train and validation split
    train_test_dataset = train_prompts['train'].train_test_split(test_size=0.2, seed=42)

    train_prompts['train'] = train_test_dataset['train']
    train_prompts['validation'] = train_test_dataset['test']

    return train_prompts['train'], train_prompts['validation'], test_data

############# DATASET FOR TRAINING AND TEST ################
data = pd.read_csv(dataset_csv_file)
train_prompts, val_prompts, test_data = split_data(data)

####### MODEL ##################
# INITIALIZE MODEL
device_string = PartialState().process_index

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # loading in 4 bit
    bnb_4bit_quant_type="nf4", # quantization type
    bnb_4bit_use_double_quant=True, # nested quantization
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Fakes fix it 
classes_to_predict = train_prompts['train']['label'].unique()
id2label = {i: label for i, label in enumerate(classes_to_predict)}
label2id = {label: i for i, label in enumerate(classes_to_predict)}

model_config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=LLM_MODEL,
    token=ACCESS_TOKEN,
)

model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=LLM_MODEL,
    config=model_config,
    quantization_config=bnb_config if quantize else None,
    device_map={'':device_string},
    token=ACCESS_TOKEN,
    num_labels = len(classes_to_predict),
    id2label = id2label,
    label2id = label2id 
)

model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=LLM_MODEL,
    trust_remote_code=True,
    token=ACCESS_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

#################### TRAINING ###################àà
def compute_metrics(pred):
    preds, labels = pred

    # Calculate accuracy precision, recall, and F1-score
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=16,
    task_type="SEQ_CLS"
)

model = get_peft_model(model, peft_config)

training_arguments = TrainingArguments(
    output_dir="./checkpoints/",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    #gradient_checkpointing_kwargs={'use_reentrant':False},
    optim="paged_adamw_32bit",
    num_train_epochs=10,
    save_steps=500,
    learning_rate=1e-4,
    warmup_steps=100,
    load_best_model_at_end=True,
    evaluation_strategy="steps",
)

collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_prompts,
    eval_dataset = val_prompts,
    peft_config = peft_config,
    data_collator=collator,
    max_seq_length=2048,
    args=training_arguments,
    dataset_text_field='text',
    compute_metrics = compute_metrics()
)

trainer.train()

# Save the model
model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
model_to_save.save_pretrained("clf")
