import os
import tempfile
from pathlib import Path
import joblib
import requests
import torch
import transformers
import xml.etree.ElementTree as ET
import re
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report

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

from peft import LoraConfig

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

ACCESS_TOKEN = "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"

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

CLASS = 'uptake'
CLASS_IN_CODEBOOK = 'Uptake'

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

class LLM:
    def __init__(self, 
                 codebook_file,
                 use_xmls = [],
                 use_websites = [],
                 use_history = True,
                 use_buffer = False,
                 quantize= False,
                 window_size= 3,
                 text_field= 'message',
                 combine_fields = [],
                 separator = ': ',
                 dataset_csv_file = None
                 ):

        self.quantize = quantize
        self.initialize_model(quantize)
        self.use_history = use_history
        self.classes = []
        self.codebook = self.get_codebook(codebook_file)
        self.window_size = window_size
        self.text_field = text_field
        self.combine_fields = combine_fields

        if len(use_xmls) != 0 or len(use_websites) != 0:
            self.initialize_retriever_chain(xml=use_xmls, web=use_websites)
        else:
            self.retriever_chain = None

        self.use_buffer = use_buffer
        if use_buffer:
            self.initialize_buffer()
        else:
            self.memory_buffer = None

        if dataset_csv_file is not None:
            self.train_data, self.test_data = self.preprocess_dataset(dataset_csv_file)
        else:
            self.train_data, self.test_data = None, None

        self.initialize_llm()
        self.llm_chain = self.get_chain()
    
    def initialize_model(self, quantize = False):
        # INITIALIZE MODEL
        model_config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name_or_path=LLM_MODEL,
            token=ACCESS_TOKEN,
        )

        if quantize:
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True, # loading in 4 bit
                bnb_4bit_quant_type="nf4", # quantization type
                bnb_4bit_use_double_quant=True, # nested quantization
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=LLM_MODEL,
                config=model_config,
                quantization_config=bnb_config, # we introduce the bnb config here.
                device_map="auto",
                token=ACCESS_TOKEN
            )
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=LLM_MODEL,
                config=model_config,
                device_map="auto",
                token=ACCESS_TOKEN
            )
        self.model.config.use_cache = False

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=LLM_MODEL,
            trust_remote_code=True,
            token=ACCESS_TOKEN,
            padding_side = "right"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def initialize_llm(self):
        self.model.eval()

        #TODO: finetune the model
        generate_text = transformers.pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            #temperature=0.0,
            max_new_tokens=500,
            repetition_penalty=1.5,
        )
        self.llm = HuggingFacePipeline(pipeline=generate_text)

    def initialize_retriever_chain(self, xml = [], web = []):
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
        self.retriever_chain = create_history_aware_retriever(
            llm=self.llm,
            retriever=retreiver,
            prompt = story_retriever_prompt,
        )

    def initialize_buffer(self):
        self.memory_buffer = ConversationBufferWindowMemory(
                memory_key="history", 
                return_messages=True, 
                output_key="answer", 
                llm=self.llm,
                k=self.window_size,
            )
    
    def format_row_codebook(self, row):
        class_message = f"Class: '''{row['Term']}'''\n"
        if row['Definition'] is not None:
            definition = " ".join(row['Definition'].split('\n'))
            class_message += f"Definition: '''{definition}'''\n"
        if row['Example'] is not None:
            example = " ".join(row['Example'].split('\n'))
            class_message += f"Example: '''{example}'''\n"
        class_message += "\n"
        return class_message

    def get_codebook(self, codebook_file):
        codebook = pd.read_excel(codebook_file)
        codebook[['Class', 'Term']] = codebook['Term'].str.split(':', expand=True)
        codebook = codebook[codebook['Class'] == CLASS_IN_CODEBOOK]
        codebook.drop(columns=['Class'], inplace=True)
        codebook['Term'] = codebook['Term'].map(lambda x: x.strip())
        self.classes = codebook['Term'].tolist()
        codebook_list = codebook.apply(self.format_row_codebook, axis=1)
        codebook_list = "".join(codebook_list)
        return codebook_list
        
    def clear_memory(self):
        if self.memory_buffer:
            self.memory_buffer.clear()

    def get_messages(self):
        if self.codebook is None:
            raise Exception("Codebook is not initialized")
        
        messages = []

        messages.append(("system", f"You are a helpful AI classifier that knows everything about discourse analysis and can help children to classify their discussion.\nClassify the following sentence into one of the following classes:\n{self.codebook}\nIf you failed to classify the sentence, return None instead of caming up with a solution.\n"))

        if self.retriever_chain is not None:
            messages.append(("system", "Here are some relevant documents that might help you to classify the sentence:\n'''\n{context}\n'''\n"))
        
        if self.use_history or self.memory_buffer is not None:
            messages.append(("system", "Here is the chat history of the children discussion:"))
            messages.append(("placeholder", "{history}"))
        
        messages.append(("system", "Classify the sentence into one class of the codebook."))
        messages.append(("human", "{input}"))
        return messages
    
    def get_chain(self):
        template = ChatPromptTemplate.from_messages(self.get_messages())
        if self.retriever_chain is not None:
            chain = create_stuff_documents_chain(self.llm, template)
        else:
            chain = template | self.llm
        return chain

    def predict_single(self, sentence, history = None):
        if self.llm_chain is None:
            raise Exception("Chain is not initialized")
        
        request = {}
        request['input'] = sentence

        if self.memory_buffer is not None and history is None:
            history = self.memory_buffer.load_memory_variables({})
            request['history'] = history['history']
        
        if self.use_history:
            if history is None:
                request['history'] = ''
            else:
                request['history'] = history
        
        if self.retriever_chain is not None:
            docs = self.retriever_chain.invoke(request)
            request['context'] = docs
        
        response = self.llm_chain.invoke(request)
        answer = find_first(response, self.classes)

        if self.memory_buffer is not None:
            self.memory_buffer.save_context(inputs={'input': sentence}, outputs={'answer': answer})
        
        return answer

    def format_requests(self, data):
        if self.use_history:
            requests = data.apply(lambda x: {'input': x['chat'], 'history': [("human", chat) for chat in x['history'].split('\n')] if not pd.isna(x['history']) else []}, axis=1)
        else:
            requests = data.apply(lambda x: {'input': x['chat'],}, axis=1)

        if self.retriever_chain is not None:
            list_docs = self.retriever_chain.batch(requests)
            for i in range(len(requests)):
                requests[i]['context'] = list_docs[i]
        
        return requests

    def predict_batch(self, data):
        if self.llm_chain is None:
            raise Exception("Chain is not initialized")
        
        requests = self.format_requests(data)
        
        responses = self.llm_chain.batch(requests)

        answers = []
        for response in responses:
            print(response)
            answer = find_first(response, self.classes)
            answers.append(answer)

        return answers

    def preprocess_dataset(self, dataset_csv_file):
        data = pd.read_csv(dataset_csv_file)
        
        data[self.text_field] = data[self.combine_fields].apply(lambda x: self.separator.join(x.dropna().astype(str)), axis=1)

        history = []
        for i in range(len(data)):
            if i >= 1 and not data.iloc[i][['book_id', 'bookclub', 'course']].equals(data.iloc[i-1][['book_id', 'bookclub', 'course']]):
                history = []

            data.at[i, 'history'] = '\n'.join(history) if history else pd.NA

            history.append(data.iloc[i][self.text_field])
            if len(history) > self.window_size:
                history.pop(0)

        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        return train_data, test_data

    def get_trainable_dataset(self, data):
        requests = data.apply(lambda x: {'input': x[self.text_field], 'history': [("human", chat) for chat in x['history'].split('\n')] if not pd.isna(x['history']) else []}, axis = 1)
        messages = self.get_messages()
        prompt_template = ChatPromptTemplate.from_messages(messages)
        prompts = prompt_template.batch(list(requests))
        prompts = list(map(lambda x: x.to_string(), prompts))

        dataset_dict = DatasetDict({
            'train': Dataset.from_dict({'text': prompts, 'label': data[CLASS]})
        })

        # train, validation and test split
        train_test_dataset = dataset_dict['train'].train_test_split(test_size=0.2, seed=42)

        dataset_dict['train'] = train_test_dataset['train']
        dataset_dict['validation'] = train_test_dataset['test']
        return dataset_dict

    def train(self):
        if self.train_data is None:
            raise Exception("No dataset found, please add dataset while initializing.")

        dataset = self.get_trainable_dataset(self.train_data)

        peft_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.1,
            r=16,
            bias="none",
            task_type="CAUSAL_LM"
        )

        training_arguments = transformers.TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=4,
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
        )

        trainer.train()

    def test(self):
        y_test = self.test_data[CLASS]
        y_pred = self.predict_batch(self.test_data)
        
        print('Accuracy:', accuracy_score(y_test, y_pred))
        print('Precision:', precision_score(y_test, y_pred, average='weighted', zero_division=0))
        print('Recall:', recall_score(y_test, y_pred, average='weighted', zero_division=0))
        print('F1:', f1_score(y_test, y_pred, average='weighted', zero_division=0))

        print(classification_report(y_test, y_pred, zero_division=0))


# Initialize the model
my_llm_classifier = LLM(codebook_file = './data/codebook.xlsx',
                        use_xmls = [],#['./data/LadyOrThetigerIMapBook.xml'],
                        use_websites = [],
                        use_history = True,
                        use_buffer = False,
                        quantize=False,
                        window_size= 3,
                        text_field= 'message',
                        combine_fields = [],
                        separator = ': ',
                        dataset_csv_file= '/data/cleaned_data.csv'
                        )