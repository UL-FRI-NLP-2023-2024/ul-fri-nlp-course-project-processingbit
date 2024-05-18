import tempfile
import requests
import torch
from transformers import AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import xml.etree.ElementTree as ET
import re
import pandas as pd
import numpy as np
from datasets import DatasetDict, load_dataset, Dataset
from tqdm import tqdm

#from IPython.display import print
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.docstore.document import Document

from langchain_core.output_parsers import StrOutputParser

####### MODEL CONFIGURATION #######
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
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

####### FILE CONFIGURATION #######
DATASET_FILE = "./data/cleaned_data.csv"



######### PARSING FUNCTIONS ########

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

class Retriever:
    def __init__(self, xmls = [], websites = [], quantize = True, documents_to_retrieve = 1):
        # Loading documents
        docs = []
        for xml_path in xmls:
            docs += parse_xml(xml_path)
        docs += fetch_websites(websites)

        # Splitting documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        story_splitted = text_splitter.split_documents(docs)

        # Creating retriever
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cuda"},
        )
        vector_store = FAISS.from_documents(story_splitted, embeddings)
        self.retreiver = vector_store.as_retriever(
            search_type="similarity",
            k = documents_to_retrieve,
        )

        ### Initializing model, tokenizer and LLM
        # Model
        model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=LLM_MODEL,
            token=ACCESS_TOKEN,
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, # loading in 4 bit
            bnb_4bit_quant_type="nf4", # quantization type
            bnb_4bit_use_double_quant=True, # nested quantization
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=LLM_MODEL,
            config=model_config,
            quantization_config=bnb_config if quantize else None,
            device_map= "auto",
            token=ACCESS_TOKEN
        )
        model.config.use_cache = True
        model.eval()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=LLM_MODEL,
            trust_remote_code=True,
            token=ACCESS_TOKEN
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # LLM
        llm = HuggingFacePipeline(pipeline=pipeline(
            task="text-generation",
            model=model,
            tokenizer=self.tokenizer,
            return_full_text = False,
            temperature = 0.5,
            max_new_tokens = 500,
            repetition_penalty = 1.5,
            do_sample = True,
        ))

        self.get_query = llm | StrOutputParser()

        self.system_chat = [
            {
                "role": "system",
                "content": "You are an AI summarizer. You need to summarize the topic of the conversation to help another AI to retrieve the most similar document.",
            },
        ]
        
    def invoke(self, request):
        query = request['input']
        if 'chat_history' in request:
            chat = self.system_chat
            chat_history = request['chat_history'] if request['chat_history'] else ''
            for user_messsage in chat_history.split('\n'):
                chat.append({"role": "user", "content": user_messsage})
            chat.append({"role": "user", "content": request['input']})

            query = self.get_query.invoke(chat)

        return self.retreiver.invoke(query)


###### DATA PROCESSING FUNCTIONS ######
def preprocess_data(
        combine_fields = [],
        separator = ': ',
        text_field = 'message',
        history_field = 'chat_history',
        unique_keys_for_conversation =  ['book_id', 'bookclub', 'course'],
        window_size = 3,
):
    data = pd.read_csv(DATASET_FILE)
    if len(combine_fields) > 0:
        data[text_field] = data[combine_fields].apply(lambda x: separator.join(x.dropna().astype(str)), axis=1)

    history = []
    for i in range(len(data)):
        if i >= 1 and not data.iloc[i][unique_keys_for_conversation].equals(data.iloc[i-1][unique_keys_for_conversation]):
            history = []

        data.at[i, history_field] = '\n'.join(history) if history else pd.NA

        history.append(data.iloc[i][text_field])

        if window_size > 0 and len(history) > window_size:
            history.pop(0)
    return data

##### MAIN FUNCTION ######

if __name__ == '__main__':

    use_history = True
    quantize = True
    window_size = -1

    retriever = Retriever(xmls = ['./data/LadyOrThetigerIMapBook.xml'], 
                            quantize = quantize,
                            documents_to_retrieve = 1
                            )
            
    # Data processing
    text_field = 'message'
    history_field = 'history'

    data = preprocess_data(
        combine_fields = [],
        separator = ': ',
        text_field = text_field,
        history_field = history_field,
        unique_keys_for_conversation =  ['book_id', 'bookclub', 'course'],
        window_size = window_size,
    )[:2]

    # Prompt requests with input and w/wo history
    if use_history:
        prompt_requests = data.apply(lambda x: {'input': x[text_field],
                                                'chat_history': x[history_field],
                                                }, axis=1).to_list()
    else:
        prompt_requests = data.apply(lambda x: {'input': x[text_field]}, axis=1).to_list()

    
        # Retrieval
    print('Retrieving context...')
    docs_data = []
    for prompt in tqdm(prompt_requests):
        docs = retriever.invoke(prompt)
        print('Retrieved context:', docs)
        print('number of docs:', len(docs))

        docs_data.append(docs)

    dataset = pd.DataFrame(docs_data)

    # Save dataset
    dataset.to_csv(f"./preprocessed/context.csv", index=False)
