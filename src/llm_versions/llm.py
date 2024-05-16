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

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

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


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    story = ''
    for page in root.findall('.//page'):
        page_type_id = page.attrib['type_id']
        state_text = page.find('state/text').text.strip()
        story += state_text + '\n'

    story = story.replace('<p>', '')
    story = story.replace('</p>', '\n')
    story = re.sub(r'\n+', '\n', story)
    story = story.strip()
    story_doc = [Document(page_content=story)]
    return story_doc

def fetch_websites(sites):
    docs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/site.html"
        for site in sites:
            res = requests.get(site)
            with open(filename, mode="wb") as fp:
                fp.write(res.content)
            docs.extend(BSHTMLLoader(filename, open_encoding = 'utf8').load())
    return docs

def compute_metrics(pred, classes = None):
    responses, labels = pred

    preds = [find_first(response) for response in responses]
    print('ciao')

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

def get_messages(use_history,
                 use_context,
                     ):
    messages = []
    ## Initial
    messages.append(("system", "You are an AI expert in categorizing sentences."))

    ### Final input
    messages.append(("system", "Categorize the following sentence into one class of the codebook."))
    messages.append(("human", "{input}"))
    messages.append(("system", "Remember to answer with only the name of the class and nothing else.\
If you failed to categorize the sentence, don't answer it, but return None."))
    
    messages.append(("assistant", "### IMPORTANT CODEBOOK:\n\
{codebook}\
###"))

    ### Context
    if use_context:
        messages.append(("assistant", "Here are some relevant documents that might help you to classify the sentence:\n\
'''\n\
{context}\n\
'''\n\
"))
    
    ### History
    if use_history:
        messages.append(("assistant", "You can use the following chat history if it is relevant:"))
        messages.append(("placeholder", "{history}"))
    
    messages.append(("assistant", "### Answer:"))

    return messages

class Codebook:
    def __init__(self,
                 codebook_excel_file) -> None:
        
        codebook = pd.read_excel(codebook_excel_file)
        codebook[['Class', 'Term']] = codebook['Term'].str.split(':', expand=True)
        codebook['Term'] = codebook['Term'].map(lambda x: x.strip())
        
        self.codebook = codebook

    def format_row_codebook(self, row):
        class_message = f"Class: '''{row['Term']}'''\n"

        if not pd.isna(row['Definition']):
            definition = " ".join(row['Definition'].split('\n'))
            class_message += f"Definition: '''{definition}'''\n"

        if not pd.isna(row['Example']):
            example = " ".join(row['Example'].split('\n'))
            class_message += f"Example: '''{example}'''\n"

        class_message += "\n"
        return class_message

    def get_codebook_of_class(self, class_to_predict):
        new_codebook = self.codebook[self.codebook['Class'] == class_to_predict].copy()
        new_codebook.drop(columns=['Class'], inplace=True)
        return new_codebook

    def get_classes_of(self, class_to_predict):
        return self.get_codebook_of_class(class_to_predict)['Term'].to_list()

    def get_classes_to_predict(self):
        return self.codebook['Class'].to_list()

    def format_codebook(self, class_to_predict):
        new_codebook = self.get_codebook_of_class(class_to_predict)
        codebook_list = new_codebook.apply(self.format_row_codebook, axis=1)
        codebook_list = "".join(codebook_list)
        return codebook_list
    

class LLM:
    def __init__(self, 
                 codebook : Codebook,
                 use_xmls = [],
                 use_websites = [],
                 use_history = True,
                 use_buffer = False,
                 quantize= False,
                
                 dataset_csv_file = None,
                 window_size= 3,
                 text_field= 'message',
                 history_field = 'history',
                 combine_fields = [],
                 separator = ': ',
                 unique_keys_for_conversation : list = ['book_id', 'bookclub', 'course'],
                 
                 use_adapters = False,
                 ):
        self.use_history = use_history
        self.use_buffer = use_buffer
        self.use_context = (len(use_xmls) != 0 or len(use_websites) != 0)

        self.template = ChatPromptTemplate.from_messages(get_messages(
            use_context= self.use_context,
            use_history= self.use_history
        ))

        ### CODEBOOK
        self.codebook = codebook

        ### DATASET ARGUMENTS
        self.window_size = window_size
        self.text_field = text_field
        self.history_field = history_field
        self.combine_fields = combine_fields
        self.separator = separator
        self.unique_keys_for_conversation = unique_keys_for_conversation

        ### DATASET PREPROCESSING AND SPLITTING
        if dataset_csv_file is not None:
            self.train_data, self.test_data = self.preprocess_dataset(dataset_csv_file)
        else:
            self.train_data, self.test_data = None, None

        ### MODEL SPECIFICATIONS
        self.quantize = quantize
        self.use_adapters = use_adapters

        self.model = self._build_model()
        self.tokenizer = self._build_tokenizer()
        self.llm = self._build_llm()

        ### RETRIEVER FOR CONTEXT
        if self.use_context:
            self.retriever_chain = self._build_retriever_chain(xml=use_xmls, web=use_websites)
        else:
            self.retriever_chain = None

        ### HISTORY BUFFER AND NORMAL HISTORY SETTING
        if use_buffer:
            self.memory_buffer = ConversationBufferWindowMemory(
                memory_key="history", 
                return_messages=True, 
                output_key="answer", 
                llm=self.llm,
                k=self.window_size,
            )
        else:
            self.memory_buffer = None

        ### BUILD CHAIN
        self.llm_chain = self._build_chain()
    
    def _build_model(self):
        # INITIALIZE MODEL
        model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=LLM_MODEL,
            token=ACCESS_TOKEN,
        )
        device_string = PartialState().process_index
        if self.quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, # loading in 4 bit
                bnb_4bit_quant_type="nf4", # quantization type
                bnb_4bit_use_double_quant=True, # nested quantization
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=LLM_MODEL,
                config=model_config,
                quantization_config=bnb_config, # we introduce the bnb config here.
                device_map={'':device_string},
                token=ACCESS_TOKEN
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=LLM_MODEL,
                config=model_config,
                device_map={'':device_string},
                token=ACCESS_TOKEN
            )
        model.config.use_cache = False
        model.eval()
        return model

    def _build_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=LLM_MODEL,
            trust_remote_code=True,
            token=ACCESS_TOKEN
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        return tokenizer

    def _build_llm(self):
        #TODO: finetune the model
        llm = HuggingFacePipeline(pipeline=pipeline(
            # ['audio-classification', 'automatic-speech-recognition', 'conversational', 'depth-estimation', 'document-question-answering', 'feature-extraction', 'fill-mask', 'image-classification', 'image-feature-extraction', 'image-segmentation', 'image-to-image', 'image-to-text', 'mask-generation', 'ner', 'object-detection', 'question-answering', 'sentiment-analysis', 'summarization', 'table-question-answering', 'text-classification', 'text-generation', 'text-to-audio', 'text-to-speech', 'text2text-generation', 'token-classification', 'translation', 'video-classification', 'visual-question-answering', 'vqa', 'zero-shot-audio-classification', 'zero-shot-classification', 'zero-shot-image-classification', 'zero-shot-object-detection', 'translation_XX_to_YY']"
            task="text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            #return_full_text=False,
            #temperature=0.0,
            max_new_tokens=500,
            repetition_penalty=1.5
        ))
        return llm

    def _build_retriever_chain(self, xml = [], web = []):
        docs = []
        for xml_path in xml:
            docs += parse_xml(xml_path)
        docs += fetch_websites(web)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        story_splitted = text_splitter.split_documents(docs)

        embedding = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cuda"},
        )
        vectorstore = FAISS.from_documents(story_splitted, embedding)
        retreiver = vectorstore.as_retriever(
            search_type="similarity",
            k=6,
        )
        story_retriever_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an helpful AI assistant that knows everything about the story in the documents.\nYou need to retrieve the most relevant document based on the following chat:"),
            ("placeholder", "{history}"),
            ("human", "{input}")
        ])

        retriever_chain =  create_history_aware_retriever(
            llm=self.llm,
            retriever=retreiver,
            prompt = story_retriever_prompt,
        )
        return retriever_chain
    
    def _build_chain(self):
        if self.llm is None:
            raise Exception('LLM not initialized')
        
        if self.retriever_chain is not None:
            chain = create_stuff_documents_chain(self.llm, self.template)
        else:
            chain = self.template | self.llm
        return chain

    def predict_single(self, sentence, class_to_predict, history = None):
        if self.llm_chain is None:
            raise Exception("Chain is not initialized")
        
        request = {}
        request['input'] = sentence
        request['codebook'] = self.codebook.format_codebook(class_to_predict)

        if history is None:
            if self.memory_buffer is None:    
                request['history'] = ''
            else:
                history = self.memory_buffer.load_memory_variables({})
                request['history'] = history['history']
        else:        
            request['history'] = history
        
        if self.retriever_chain is not None:
            docs = self.retriever_chain.invoke(request)
            request['context'] = docs
        
        response = self.llm_chain.invoke(request)
        answer = find_first(response, self.codebook.get_classes_of(class_to_predict))

        if self.memory_buffer is not None:
            self.memory_buffer.save_context(inputs={'input': sentence}, outputs={'answer': answer})
        
        return answer

    def format_requests(self, data, class_to_predict):

        formatted_codebook = self.codebook.format_codebook(class_to_predict)
        if self.use_history:
            requests = data.apply(lambda x: {'input': x[self.text_field], 'codebook': formatted_codebook, 'history': [("human", chat) for chat in x[self.history_field].split('\n')] if not pd.isna(x[self.history_field]) else []}, axis=1).to_list()
        else:
            requests = data.apply(lambda x: {'input': x[self.text_field], 'codebook': formatted_codebook}, axis=1).to_list()
        
        if self.retriever_chain is not None:
            list_docs = self.retriever_chain.batch(requests)
            for i in range(len(requests)):
                requests[i]['context'] = list_docs[i]
        
        return requests


    def set_adapter(self, class_to_predict):
        try:
            print('Trying to set adapter...')
            self.model.set_adapter(class_to_predict)
        except:
            print('Adapter not added, trying to load it...')
            config_filename = f'./adapters/{class_to_predict}'
            if not os.path.exists(config_filename):
                print('Adapter not found, training adapter...')
                self.train(class_to_predict)
                print('Finish training')

            lora_config = LoraConfig.from_pretrained(config_filename)
            self.model.add_adapter(lora_config, adapter_name= class_to_predict)
            self.model.set_adapter(class_to_predict)

        self.llm = self._build_llm()

    def predict_batch(self, data, class_to_predict):
        if self.use_adapters:
            self.set_adapter(class_to_predict)
        
        requests = self.format_requests(data, class_to_predict)
        
        responses = self.llm_chain.batch(requests)

        classes = self.codebook.get_classes_of(class_to_predict)

        answers = []
        for response in responses:
            answer = find_first(response, classes)
            print('Pred: ', response)
            print('Class: ', answer)
            answers.append(answer)

        return answers

    def preprocess_dataset(self, dataset_csv_file):
        data = pd.read_csv(dataset_csv_file)
        
        if len(self.combine_fields) > 0:
            data[self.text_field] = data[self.combine_fields].apply(lambda x: self.separator.join(x.dropna().astype(str)), axis=1)

        history = []
        for i in range(len(data)):
            if i >= 1 and not data.iloc[i][self.unique_keys_for_conversation].equals(data.iloc[i-1][self.unique_keys_for_conversation]):
                history = []

            data.at[i, self.history_field] = '\n'.join(history) if history else pd.NA

            history.append(data.iloc[i][self.text_field])
            if len(history) > self.window_size:
                history.pop(0)

        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        return train_data, test_data

    def get_trainable_dataset(self, data, class_to_predict):
        requests = self.format_requests(data, class_to_predict)
        
        prompts = self.template.batch(list(requests))
        prompts = list(map(lambda x: x.to_string(), prompts))

        labels = data[class_to_predict]
        dataset_dict = DatasetDict({
            'train': Dataset.from_dict({'text': prompts, 'label': labels})
        })

        # train, validation and test split
        train_test_dataset = dataset_dict['train'].train_test_split(test_size=0.2, seed=42)

        dataset_dict['train'] = train_test_dataset['train']
        dataset_dict['validation'] = train_test_dataset['test']
        return dataset_dict

    def train(self, class_to_predict):
        dataset = self.get_trainable_dataset(self.train_data, class_to_predict)

        classes = self.codebook.get_classes_of(class_to_predict)

        peft_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.1,
            r=16,
            bias="none",
            task_type="CAUSAL_LM"
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
            save_steps=1000,
            learning_rate=1e-4,
            warmup_steps=100,
            lr_scheduler_type="constant",
            report_to="none",
            load_best_model_at_end=True,
        )

        #response_template = "### Answer"
        #collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer = self.tokenizer)

        #model = get_peft_model(self.model, peft_config)

        trainer = SFTTrainer(
            model=self.model,
            tokenizer= self.tokenizer,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            peft_config=peft_config,
            #data_collator=collator,
            max_seq_length=2048,
            args=training_arguments,
            dataset_text_field='text',
            compute_metrics = lambda pred: compute_metrics(pred, classes=classes)
        )

        trainer.train()
        #trainer.evaluate()

        trainer.save_model(f'./adapters/{class_to_predict}')

    def test(self, class_to_predict):
        y_test = self.test_data[class_to_predict]
        y_pred = self.predict_batch(self.test_data, class_to_predict)

        print("Unique_test: ", pd.value_counts(y_test))
        print("Unique_pred: ", pd.value_counts(y_pred))
        
        print('Accuracy:', accuracy_score(y_test, y_pred))
        print('Precision:', precision_score(y_test, y_pred, average='weighted', zero_division=0))
        print('Recall:', recall_score(y_test, y_pred, average='weighted', zero_division=0))
        print('F1:', f1_score(y_test, y_pred, average='weighted', zero_division=0))

        print(classification_report(y_test, y_pred, zero_division=0))

codebook_file = './data/codebook.xlsx'
codebook = Codebook(codebook_file)

# Initialize the model
my_llm_classifier = LLM(
                        ### codebook
                        codebook = codebook,

                        ### retriever
                        use_xmls = [],#['./data/LadyOrThetigerIMapBook.xml'],
                        use_websites = [],

                        ### history
                        use_history = True,
                        use_buffer = False,

                        ### load model
                        quantize=False,

                        ### DATASET ARGS
                        dataset_csv_file= './data/cleaned_data.csv',
                        window_size= 3,
                        text_field= 'message',
                        combine_fields = [],
                        separator = ': ',
                        unique_keys_for_conversation = ['book_id', 'bookclub', 'course'],
                        use_adapters=False
                        )

class_to_predict = 'Discussion'
print('Predicting class:', class_to_predict)

my_llm_classifier.test(class_to_predict)
