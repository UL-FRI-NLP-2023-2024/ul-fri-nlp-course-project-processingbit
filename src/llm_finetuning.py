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

with open("./src//tokens/hugging_face_token.txt", "r") as file:
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

def load_data(class_to_predict, use_history, use_context):
    final_path = f'./preprocessed/dataset_{class_to_predict}'
    final_path += '_with_history' if use_history else ''
    final_path += '_with_context' if use_context else ''
    dataset = load_from_disk(final_path)
    train_val_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
    dataset['train'] = train_val_dataset['train']
    dataset['validation'] = train_val_dataset['test']
    return dataset

def get_model(model_name, quantize):
    # Model
    model_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_name,
        token=ACCESS_TOKEN,
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, # loading in 4 bit
        bnb_4bit_quant_type="nf4", # quantization type
        bnb_4bit_use_double_quant=True, # nested quantization
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        config=model_config,
        quantization_config=bnb_config if quantize else None,
        device_map= "auto",
        token=ACCESS_TOKEN
    )

    return model

def get_tokenizer(model_name):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=LLM_MODEL,
        trust_remote_code=True,
        token=ACCESS_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    return tokenizer

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_adapter = False
    quantize = False

    # Load data
    class_to_predict = 'Discussion'
    use_history = True
    use_context = False
    dataset = load_data(class_to_predict, use_history, use_context)

    # Model
    model = get_model(LLM_MODEL, quantize)

    # Tokenizer
    tokenizer = get_tokenizer(LLM_MODEL)

    # Loader
    if load_adapter:
        model.load_adapter(f"./adapters/{class_to_predict}")
    else:
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        model.config.pretraining_tp = 1
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        
        if quantize:
            model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
        )
        model = get_peft_model(model, peft_config)

        training_arguments = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            optim="paged_adamw_8bit",
            save_steps=200,
            #logging_steps=30,
            learning_rate=2e-4,
            #fp16=False,
            #bf16=False,
            max_grad_norm=0.3,
            #max_steps=-1,
            group_by_length=True,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            report_to="none"
        )

        # Setting sft parameters
        def formatting_prompts_func(example):
            output_texts = []
            #for i in range(len(example['prompt'])):
            for i in range(len(example['instruction'])):
                text = f"<s>[INST] {example['instruction'][i].replace('</div>', '').strip()}\n [/INST] {example['output'][i]}</s>"
            return output_texts

        response_template = "[/INST]"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

        trainer = SFTTrainer(
            model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            formatting_func=formatting_prompts_func,
            data_collator=collator,
            peft_config=peft_config,
            max_seq_length=2048,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=False,
        )

        # Training
        trainer.train()

        # Evaluation
        eval_results = trainer.evaluate()
        print(eval_results)

        # Save the model
        model.save_adapter(f"./adapters/{class_to_predict}")

    # Testing
    model.config.use_cache = True
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = False
    model.eval()

    llm = HuggingFacePipeline(pipeline=pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            temperature=0.2,
            max_new_tokens=500,
            repetition_penalty=1.5,
            do_sample=True,
        ))

    
    

