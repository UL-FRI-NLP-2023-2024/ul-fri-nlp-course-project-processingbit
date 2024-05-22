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

    ######### Preprocess data #########
    data = load_data_predicts(path_dir, class_to_predict, model_name)
    prep_data = preprocess_data(dataset_file='./data/cleaned_data.csv',
                                    class_field='Discussion')
    
    # Merge the data
    #left_data = data.drop(columns=['text'])
    results = pd.merge(data, prep_data[['message']], left_on='index', right_index=True)
    
    classes = results['labels'].unique()
    
    model_types = results.columns.to_list()
    model_types = [model for model in model_types if model not in ['index', 'message', 'labels', 'text']]

    initial_prompt = "You are an human ensemble. You have to predict the class of the following text, based on the results of other models.\n"
    initial_prompt += "You will receive some information about the classes, the text and the other models predictions.\n"

    prompts = []
    for index, row in results.iterrows():
        row['text'][0] = initial_prompt +

        
        #for message in row['text']:
        #    prompt += f"[{message['content']}]\n"



        prompt += "'''\n"
        prompt += "The last input sentence is the sentence that the models are trying to classify.\n"
        prompt += "Now, the other models have predicted the following classes:\n"

        for model in model_types:
            prompt += f"Model {model} predicted {row[model]},\n"

        prompt += "What class do you this is the correct one? Answer with only the class that you think it's right.\n"
        
        prompts.append(prompt)

    # Load the model
    model = get_model(llm_model, quantize=True)
    tokenizer = get_tokenizer(llm_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.config.use_cache = True
    model.eval()

    pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            #temperature=0.5,
            max_new_tokens=100,
            repetition_penalty=1.5,
            #do_sample=True,
        )

    generated_texts = pipe(prompts)

    answers = [answer[0]['generated_text'] for answer in generated_texts]

    post_processed_answers = [find_first(answer, classes) for answer in answers]

    for i, class_answer in enumerate(post_processed_answers):
        print('Generated:', answers[i])
        print('Post-processed:', class_answer)
        print('Real Label:', results['labels'][i])
        print()

    # Classification report
    print(classification_report(results['labels'], post_processed_answers))





    

