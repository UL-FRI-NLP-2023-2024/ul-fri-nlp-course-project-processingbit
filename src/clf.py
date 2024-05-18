import os
import tempfile
from pathlib import Path
import joblib
import requests
import torch
from transformers import AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM 
from transformers import AutoTokenizer, pipeline, TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer, TrainerCallback, EarlyStoppingCallback

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
from datasets import load_from_disk

from peft import LoraConfig, PeftConfig, get_peft_model

test_name = "Input_4096_with_no_context"

print("#### NAME #####")
print(test_name)

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

quantize = True
dataset_file = './preprocessed/dataset_Discussion_with_history'
text_field = 'text'

############# DATASET FOR TRAINING AND TEST ################
data_with_test = load_from_disk(dataset_file)
data = data_with_test['train'].train_test_split(test_size=0.2, seed=42)

print(data.shape)

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
classes_to_predict = np.unique(data['train']['labels'])
id2label = {i: label for i, label in enumerate(classes_to_predict)}
label2id = {label: i for i, label in enumerate(classes_to_predict)}

model_config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=LLM_MODEL,
    token=ACCESS_TOKEN,
    num_labels = len(classes_to_predict),
)

model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=LLM_MODEL,
    config=model_config,
    quantization_config=bnb_config if quantize else None,
    device_map={'':device_string},
    token=ACCESS_TOKEN,
    #id2label = id2label,
    #label2id = label2id 
)

model.config.use_cache = False

print("#### GET TOKENIZER #####")
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=LLM_MODEL,
    trust_remote_code=True,
    token=ACCESS_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

#################### TRAINING ###################àà
def compute_metrics(pred):
    preds, labels = pred.predictions, pred.label_ids

    preds = np.argmax(preds, axis=1)
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
    lora_alpha=64,
    lora_dropout=0.1,
    r=64,
    #target_modules="all-linear",
    task_type="SEQ_CLS"
)

print("#### GET PEFT #####")
model = get_peft_model(model, peft_config)
model.config.pad_token_id = model.config.eos_token_id

training_arguments = TrainingArguments(
    output_dir="./checkpoints/",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    num_train_epochs=50,
    logging_steps=20,
    save_steps=20,
    learning_rate=1e-4,
    warmup_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    evaluation_strategy="steps",
)

print("#### TOKENIZE DATA #####")
max_length = 4096
def preprocess_function(examples):
    global text_field
    if isinstance(text_field, str):
        d = examples[text_field]
    else:
        d = examples[text_field[0]]
        for n in text_field[1:]:
            nd = examples[n]
            assert len(d) == len(nd)
            for i, t in enumerate(nd):
                d[i] += '\n' + t

    return tokenizer(d, padding='longest', max_length=max_length, truncation=True)


tokenized_data = data.map(preprocess_function, batched=True)

def encode_labels(example):
    example['labels'] = label2id[example['labels']]
    return example

tokenized_data = tokenized_data.map(encode_labels)
collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = tokenized_data['train'].shard(index=1, num_shards=10),
    eval_dataset = tokenized_data['test'],
    data_collator=collator,
    args=training_arguments,
    compute_metrics = compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("#### TRAIN ####")
print(trainer.evaluate())

trainer.train()

print("#### SAVE ####")
# Save the model
model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
model_to_save.save_pretrained("clf-" + test_name)