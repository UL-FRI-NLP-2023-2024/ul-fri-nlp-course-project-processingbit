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
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report

#from IPython.display import print
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain_core.memory import BaseMemory
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.docstore.document import Document

from peft import LoraConfig

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

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    story = ''
    for page in root.findall('.//page'):
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

def find_first(sentence, items):
    first = len(sentence)
    item = None
    for i in items:
        index = sentence.find(i)
        if index != -1 and index < first:
            first = index
            item = i
    if item is None:
        item = 'None'
    return item

def compute_metrics(responses, labels, classes = None):
    preds = list(map(lambda response: find_first(response, classes), responses))
    
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
    ### Initial
    messages.append(("system", "You are a helpful AI classifier that knows everything about discourse analysis and can help children to classify their discussion.\n"))
    
    ### Codebook
    messages.append(("system", "Classify the new sentence into one of the classes from the codebook:\n### CODEBOOK:\n{codebook}\n###\nIf you failed to classify the sentence, return None instead of caming up with a solution.\n"))

    ### Context
    if use_context:
        messages.append(("assistant", "Here are some relevant documents that might help you to classify the sentence:\n'''\n{context}\n'''\n"))
    
    ### History
    if use_history:
        messages.append(("placeholder", "{history}"))
    
    ### Final input
    messages.append(("user", "Classify the sentence into one class of the codebook."))
    messages.append(("user", "{input}"))
    messages.append(("user", "answer:"))
    return messages

def format_requests(data, 
                        formatted_codebook,
                        format_history = False,
                        retriever_chain = None,
                        text_field = 'message',
                        history_field = 'history'
                        ):
    if format_history:
        requests = data.apply(lambda x: {'input': x[text_field], 'codebook': formatted_codebook, 'history': [("human", chat) for chat in x[history_field].split('\n')] if not pd.isna(x[history_field]) else []}, axis=1).to_list()
    else:
        requests = data.apply(lambda x: {'input': x[text_field], 'codebook': formatted_codebook,}, axis=1).to_list()
    
    if retriever_chain is not None:
        list_docs = retriever_chain.batch(requests)
        for i in range(len(requests)):
            requests[i]['context'] = list_docs[i]

    return requests


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
        new_codebook = self.codebook[self.codebook['Class'] == class_to_predict]
        new_codebook.drop(columns=['Class'], inplace=True)
        return new_codebook

    def get_classes(self, class_to_predict):
        return self.get_codebook_of_class(class_to_predict)['Term'].to_list()

    def format_codebook(self, class_to_predict):
        new_codebook = self.get_codebook_of_class(class_to_predict)
        codebook_list = new_codebook.apply(self.format_row_codebook, axis=1)
        codebook_list = "".join(codebook_list)
        return codebook_list

class Custom_Dataset:
    def __init__(self, 
                 dataset_csv_file,
                 window_size= 3,
                 text_field= 'message',
                 history_field = 'history',
                 combine_fields = [],
                 unique_keys_for_conversation : list = ['book_id', 'bookclub', 'course'],
                 separator = ': ',
                 train_size = 0.8
                 ) -> None:
        
        self.text_field = text_field
        self.history_field = history_field

        data = pd.read_csv(dataset_csv_file)
        if combine_fields and len(combine_fields) >=1 :
            data[text_field] = data[combine_fields].apply(lambda x: separator.join(x.dropna().astype(str)), axis=1)

        history = []
        for i in range(len(data)):
            if i >= 1 and not data.iloc[i][unique_keys_for_conversation].equals(data.iloc[i-1][unique_keys_for_conversation]):
                history = []
            data.at[i, 'history'] = '\n'.join(history) if history else pd.NA
            history.append(data.iloc[i][text_field])
            if len(history) > window_size:
                history.pop(0)

        self.train_data, self.test_data = train_test_split(data, train_size=train_size, random_state=42)

    def get_train_data(self, 
                        formatted_codebook,
                        use_history,
                        retriever_chain,
                        eval_size = 0.2
                    ):
        
        requests = format_requests(self.train_data, 
                                   formatted_codebook,
                                   format_history = use_history,
                                   retriever_chain = retriever_chain,
                                    text_field = self.text_field,
                                    history_field= self.history_field
                                   )
        messages = get_messages(
            use_context= (retriever_chain is not None),
            use_history= use_history
        )
        prompt_template = ChatPromptTemplate.from_messages(messages)
        prompts = prompt_template.batch(list(requests))
        prompts = list(map(lambda x: x.to_string(), prompts))

        dataset_dict = DatasetDict({
            'train': Dataset.from_dict({'text': prompts, 'label': self.train_data[class_to_predict]})
        })

        # train, validation and test split
        train_test_dataset = dataset_dict['train'].train_test_split(test_size= eval_size, seed=42)

        dataset_dict['train'] = train_test_dataset['train']
        dataset_dict['validation'] = train_test_dataset['test']
        return dataset_dict
    
    def get_test_data(self):
        return self.test_data

def get_model(quantize = False):
    # INITIALIZE MODEL
    model_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=LLM_MODEL,
        token=ACCESS_TOKEN,
    )
    if quantize:
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
            device_map="auto",
            token=ACCESS_TOKEN
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=LLM_MODEL,
            config=model_config,
            device_map="auto",
            token=ACCESS_TOKEN
        )
    model.config.use_cache = False
    return model

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=LLM_MODEL,
        trust_remote_code=True,
        token=ACCESS_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def get_llm(model, 
            tokenizer):
    model.eval()

    #TODO: finetune the model
    generate_text = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        #temperature=0.0,
        max_new_tokens=500,
        repetition_penalty=1.5,
    )
    llm = HuggingFacePipeline(pipeline=generate_text)
    return llm

def get_llm_chain(llm,
                  messages,
                  use_retriever):
    template = ChatPromptTemplate.from_messages(messages)
    if use_retriever:
        chain = create_stuff_documents_chain(llm, template)
    else:
        chain = template | llm

    return chain

class LLM_classifier:
    def __init__(self, 
                 codebook : Codebook,
                 historical_buffer : BaseMemory = None,
                 context_from_xmls = [],
                 context_from_websites = [],
                 format_history : bool = True,
                 quantize : bool = False,
                 ):
        
        self.codebook = codebook
        self.memory_buffer = historical_buffer

        self.format_history = format_history

        self.model = get_model(quantize)
        self.tokenizer = get_tokenizer()
        self.llm = self._build_llm()

        self.use_history = self.format_history or (self.memory_buffer is not None)

        if len(context_from_xmls) != 0 or len(context_from_websites) != 0:
            self.retriever_chain = self._build_retriever_chain(xml=context_from_xmls, web=context_from_websites)
            self.use_context = True
        else:
            self.retriever_chain = None
            self.use_context = False
        
        self.chain = self._build_chain()

    def _build_llm(self):
        if self.model is None:
            raise Exception('Model not initialized')
        if self.tokenizer is None:
            raise Exception('Tokenizer not initialized')
        return get_llm(self.model,
                       self.tokenizer)

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
        story_retriever_prompt = PromptTemplate.from_messages([
            ("system", "You are an helpful AI assistant that knows everything about the story in the documents.\nYou need to retrieve the most relevant document based on the following chat:"),
            ("placeholder", "{history}"),
        ])

        return create_history_aware_retriever(
            llm=self.llm,
            retriever=retreiver,
            prompt = story_retriever_prompt,
        )
    
    def _build_chain(self):
        if self.llm is None:
            raise Exception('LLM not initialized')
        messages = get_messages(
            use_context= self.use_context,
            use_history= self.use_history
        )
        return get_llm_chain(llm = self.llm,
                             messages = messages,
                             use_retriever = self.use_context,
        )


    def predict_single(self, sentence, class_to_predict, history = None):
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
        
        response = self.chain.invoke(request)
        answer = find_first(response, self.codebook.get_classes(class_to_predict))

        if self.memory_buffer is not None:
            self.memory_buffer.save_context(inputs={'input': sentence}, outputs={'answer': answer})
        
        return answer

    def predict_batch(self, data, class_to_predict, text_field, history_field = 'history'):
        if self.chain is None:
            raise Exception("Chain is not initialized")
        
        formatted_codebook = self.codebook.format_codebook(class_to_predict)
        requests = format_requests(data, 
                                   formatted_codebook=formatted_codebook,
                                   format_history= self.format_history,
                                   retriever_chain= self.retriever_chain,
                                   text_field= text_field,
                                   history_field=history_field
                                   )

        responses = self.chain.batch(requests)

        classes = self.codebook.get_classes(class_to_predict)

        answers = list(map(lambda response: find_first(response, classes), responses))
        for i in range(len(responses)):
            print('Pred: ', responses[i])
            print('Class: ', answers[i])

        return answers

    def train_adapter(self, data : Custom_Dataset,
                       class_to_predict,
                       eval_size = 0.2):
        
        formatted_codebook = self.codebook.format_codebook(class_to_predict)

        dataset = data.get_train_data(
            formatted_codebook= formatted_codebook,
            use_history= self.use_history,
            retriever_chain= self.retriever_chain,
            eval_size=eval_size
        )

        classes = self.codebook.get_classes(class_to_predict)

        peft_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.1,
            r=16,
            bias="none",
            task_type="CAUSAL_LM"
        )

        training_arguments = TrainingArguments(
            output_dir="./checkpoints/",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            per_gpu_train_batch_size=8,
            per_gpu_eval_batch_size=8,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            save_steps=100,
            logging_steps=10,
            learning_rate=2e-4,
            fp16=True,
            max_grad_norm=0.3,
            max_steps=500,
            warmup_ratio=0.3,
            lr_scheduler_type="constant",
            report_to="none"
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            peft_config=peft_config,
            max_seq_length=2024,
            args=training_arguments,
            dataset_text_field= "text",
            compute_metrics = lambda pred: compute_metrics(pred.predictions, pred.label_ids, classes=classes)
        )

        trainer.train()

def test(llm_classifier : LLM_classifier,
            data : Custom_Dataset,
            class_to_predict,
            ):
    test_data = data.get_test_data()
    text_field = data.text_field
    history_field = data.history_field

    y_test = test_data[class_to_predict]
    y_pred = llm_classifier.predict_batch(test_data, class_to_predict, text_field, history_field)
    
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred, average='weighted', zero_division=0))
    print('Recall:', recall_score(y_test, y_pred, average='weighted', zero_division=0))
    print('F1:', f1_score(y_test, y_pred, average='weighted', zero_division=0))

    print(classification_report(y_test, y_pred, zero_division=0))


# Initialize the codebook
codebook = Codebook(codebook_excel_file='./data/codebook.xlsx')

memory_buffer = ConversationBufferWindowMemory(
                memory_key="history", 
                return_messages=True, 
                output_key="answer",
                k=3,
            )

my_llm_classifier = LLM_classifier(
    codebook,
    historical_buffer = None, # memory_buffer
    context_from_xmls = [],#['./data/LadyOrThetigerIMapBook.xml'],
    context_from_websites = [],
    format_history = True,
    quantize = False)

dataset_csv_file = './data/cleaned_data.csv'

dataset = Custom_Dataset(
    dataset_csv_file,
    window_size = 3, 
    text_field = 'message', 
    combine_fields = [], 
    unique_keys_for_conversation = ['book_id', 'bookclub', 'course'], 
    separator = ': ',
    train_size = 0.6,
    eval_size = 0.2,
    test_size = 0.2
)

test(
    my_llm_classifier,
    dataset,
    class_to_predict='Discussion'
)