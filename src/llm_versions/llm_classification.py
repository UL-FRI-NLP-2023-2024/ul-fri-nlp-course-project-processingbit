import os
import tempfile
from pathlib import Path
import joblib
import requests
import torch
from transformers import AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline, TrainingArguments, AutoModelForSequenceClassification
import xml.etree.ElementTree as ET
import re
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from accelerate import PartialState

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

from peft import LoraConfig, PeftConfig, get_peft_model

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

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
print(f'Model: {LLM_MODEL}')

class_to_predict = 'Discussion'

#use_xmls = ['./data/LadyOrThetigerIMapBook.xml'],
#use_websites = []
#use_context = (len(use_xmls) != 0 or len(use_websites) != 0)

use_history = True
quantize = False

dataset_csv_file =  './data/cleaned_data.csv'
window_size = 3
text_field = 'message'
history_field = 'history'
combine_fields = []
separator = ': '
unique_keys_for_conversation = ['book_id', 'bookclub', 'course']

use_adapters = False

######### CODEBOOK ############
codebook_excel_file = './data/codebook.xlsx'
codebook = pd.read_excel(codebook_excel_file)
codebook[['Class', 'Term']] = codebook['Term'].str.split(':', expand=True)
codebook['Term'] = codebook['Term'].map(lambda x: x.strip())

all_classes = codebook['Class'].to_list()

codebook = codebook[codebook['Class'] == class_to_predict]
codebook.drop(columns=['Class'], inplace=True)
  
def format_row_codebook(row):
    class_message = f"Class: '''{row['Term']}'''\n"
    if not pd.isna(row['Definition']):
        definition = " ".join(row['Definition'].split('\n'))
        class_message += f"Definition: '''{definition}'''\n"
    if not pd.isna(row['Example']):
        example = " ".join(row['Example'].split('\n'))
        class_message += f"Example: '''{example}'''\n"
    class_message += "\n"
    return class_message


classes_to_predict = codebook['Term'].to_list()
sorted(classes_to_predict)

# Create id2label dictionary
id2label = {i: label for i, label in enumerate(classes_to_predict)}

# Create label2id dictionary
label2id = {label: i for i, label in enumerate(classes_to_predict)}

formatted_codebook = codebook.apply(format_row_codebook, axis=1)
formatted_codebook = "".join(formatted_codebook)

####### PROMPT ##########
messages = []
messages.append(("system", "You are an AI expert in categorizing sentences."))
messages.append(("system", "Categorize the following sentence into one class of the codebook."))
messages.append(("human", "{input}"))
messages.append(("system", "Remember to answer with only the name of the class and nothing else.\
If you failed to categorize the sentence, don't answer it, but return None."))
    
messages.append(("assistant", "### IMPORTANT CODEBOOK:\n\
{codebook}\
###"))

# NO CONTEXT YET
#if use_context:
#    messages.append(("assistant", "Here are some relevant documents that might help you to classify the sentence:\n\
#'''\n\
#{context}\n\
#'''\n\
#"))
    
if use_history:
    messages.append(("assistant", "You can use the following chat history if it is relevant:"))
    messages.append(("placeholder", "{history}"))

messages.append(("assistant", "### Answer:"))

template = ChatPromptTemplate.from_messages(messages)


############# DATASET FOR TRAINING AND TEST ################

data = pd.read_csv(dataset_csv_file)
if len(combine_fields) > 0:
    data[text_field] = data[combine_fields].apply(lambda x: separator.join(x.dropna().astype(str)), axis=1)

if use_history:
    history = []
    for i in range(len(data)):
        if i >= 1 and not data.iloc[i][unique_keys_for_conversation].equals(data.iloc[i-1][unique_keys_for_conversation]):
            history = []

        data.at[i, history_field] = '\n'.join(history) if history else pd.NA

        history.append(data.iloc[i][text_field])
        if len(history) > window_size:
            history.pop(0)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Training dataset

if use_history:
    prompt_requests = train_data.apply(lambda x: {'input': x[text_field], 'codebook': formatted_codebook, 'history': [("human", chat) for chat in x[history_field].split('\n')] if not pd.isna(x[history_field]) else []}, axis=1).to_list()
else:
    prompt_requests = train_data.apply(lambda x: {'input': x[text_field], 'codebook': formatted_codebook}, axis=1).to_list()
        
prompts = template.batch(list(prompt_requests))
prompts = list(map(lambda x: x.to_string(), prompts))

labels = train_data[class_to_predict]
train_prompts = DatasetDict({
    'train': Dataset.from_dict({'text': prompts, 'label': labels})
})

# train and validation split
train_test_dataset = train_prompts['train'].train_test_split(test_size=0.2, seed=42)

train_prompts['train'] = train_test_dataset['train']
train_prompts['validation'] = train_test_dataset['test']

####### MODEL ##################
# INITIALIZE MODEL
model_config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=LLM_MODEL,
    token=ACCESS_TOKEN,
)

device_string = PartialState().process_index

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # loading in 4 bit
    bnb_4bit_quant_type="nf4", # quantization type
    bnb_4bit_use_double_quant=True, # nested quantization
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=LLM_MODEL,
    config=model_config,
    quantization_config=bnb_config if quantize else None,
    device_map={'':device_string},
    token=ACCESS_TOKEN,
    #num_labels = len(classes_to_predict),
    #id2label = id2label,
    #label2id = label2id 
)

model.config.use_cache = False
#model.eval()

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

    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)

   # Calculate precision, recall, and F1-score
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
    bias="none",
    task_type="SEQ_CLS"
)

training_arguments = TrainingArguments(
    output_dir="./checkpoints/",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing_kwargs={'use_reentrant':False},
    optim="paged_adamw_32bit",
    num_train_epochs=10,
    #fp16=True,
    save_steps=500,
    learning_rate=1e-4,
    warmup_steps=100,
    lr_scheduler_type="constant",
    report_to="none",
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    save_strategy="steps",
)

#response_template = "### Answer"
#collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer = self.tokenizer)

#model = get_peft_model(self.model, peft_config)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_prompts['train'],
    eval_dataset = train_prompts['validation'],
    peft_config = peft_config,
    #data_collator=collator,
    max_seq_length=2048,
    args=training_arguments,
    dataset_text_field='text',
    compute_metrics = compute_metrics()
)

trainer.train()

