import os
import tempfile
from pathlib import Path
import joblib
import requests
import torch
from transformers import AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline, TrainingArguments
from transformers import EarlyStoppingCallback
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
    final_path = f'./preprocessed/dataset{get_extension(class_to_predict, use_history, use_context)}'
    dataset = load_from_disk(final_path)
    # TODO higher test size
    train_val_dataset = dataset['train'].train_test_split(test_size=0.05, seed=42)
    dataset['train'] = train_val_dataset['train']
    dataset['validation'] = train_val_dataset['test']
    return dataset

def get_model(model_name, quantize):
    # Model
    device_string = PartialState().process_index

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
        device_map= {'':device_string},
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
    tokenizer.padding_side = "left"
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    return tokenizer

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_adapter = True
    load_adapter = False
    quantize = True
    max_length = 100

    # Load data
    class_to_predict = 'Discussion'
    use_history = True
    use_context = False

    dataset = load_data(class_to_predict, use_history, use_context)
    classes = np.unique(dataset['test']['labels'])

    # Model
    model = get_model(LLM_MODEL, quantize)

    # Tokenizer
    tokenizer = get_tokenizer(LLM_MODEL)

    # Loader
    if use_adapter:
        if load_adapter:
            model.load_adapter(f"./adapters/mistral{get_extension(class_to_predict, use_history, use_context)}")
        else:
            model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
            model.config.pretraining_tp = 1
            
            if quantize:
                model = prepare_model_for_kbit_training(model)
            
            peft_config = LoraConfig(
                r=32,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
            )
            model = get_peft_model(model, peft_config)

            training_arguments = TrainingArguments(
                output_dir="./results",
                num_train_epochs=10,
                
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                #auto_find_batch_size=True, # FIND THE BEST BATCH SIZE THAT FEEDS THE GPU

                gradient_accumulation_steps=2,
                gradient_checkpointing_kwargs={'use_reentrant': False},
                optim="paged_adamw_8bit",

                save_strategy='steps',
                save_steps=100,

                evaluation_strategy='steps',
                eval_accumulation_steps=4,
                eval_steps=50,

                learning_rate=2e-5,
                #fp16=False,
                #max_grad_norm=0.3,
                #max_steps=-1,
                group_by_length=True,
                lr_scheduler_type="linear", # "linear", "cosine"
                warmup_ratio=0.1,
                report_to="none",

                load_best_model_at_end=True,
                metric_for_best_model="f1",
            )

            # Setting sft parameters
            def formatting_prompts_func(example):
                output_texts = []
                for i in range(len(example['text'])):
                    text = f"<s>[INST] {example['text'][i].strip()}\n [/INST] {example['labels'][i]}</s>"
                    output_texts.append(text)
                return output_texts

            def compute_metrics(pred):
                label_ids = pred.label_ids
                pred_ids = pred.predictions.argmax(-1)

                pred_ids[pred_ids == -100] = tokenizer.pad_token_id
                pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
                label_ids[label_ids == -100] = tokenizer.pad_token_id
                label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

                preds = [find_first(text, classes) for text in pred_str]
                labels = label_str
                return {
                    'accuracy': accuracy_score(labels, preds),
                    'precision': precision_score(labels, preds, average='macro'),
                    'recall': recall_score(labels, preds, average='macro'),
                    'f1': f1_score(labels, preds, average='macro')
                }

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
                compute_metrics=compute_metrics,
                callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
            )

            # Training
            trainer.train()

            # Evaluation
            eval_results = trainer.evaluate()
            print(eval_results)

            # Save the model
            #model.save_model(f"./adapters/mistral{get_extension(class_to_predict, use_history, use_context)}")

            trainer.model.save_pretrained(f"./adapters/mistral{get_extension(class_to_predict, use_history, use_context)}")

    # Testing
    model.config.use_cache = True
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = False
    model.eval()

    # Testing with dataset['test]
    inputs = tokenizer(dataset['test']['text'], return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
    outputs = model.generate(**inputs, 
                            max_length=max_length, 
                            num_return_sequences=1, 
                            do_sample=True, 
                            temperature=0.9)

    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    answers = []
    for text in generated_text:
        answer = find_first(text, classes)
        answers.append(answer)
        print('Generated:', text)
        print('Answer:', answer)

    # Classification report
    print(classification_report(dataset['test']['labels'], answers))

    # Save the generated text
    np.save(f'./results/mistral{get_extension(class_to_predict, use_history, use_context)}.npy', generated_text)
    
    

