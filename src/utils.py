import tempfile
import requests
import torch
from transformers import AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import xml.etree.ElementTree as ET
import re
import pandas as pd
import numpy as np
from datasets import DatasetDict, load_dataset, Dataset, load_from_disk
from tqdm import tqdm

# Spelling
from spellchecker import SpellChecker
from nltk.stem import  PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize

#from IPython.display import print
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.docstore.document import Document

from accelerate import PartialState

#############################################################
#############           MODEL FUNCTIONS             #########
#############################################################
def get_access_token():
    with open("./tokens/hugging_face_token.txt", "r") as file:
        access_token = file.read().strip()
    return access_token

models = {
        "gemma": "google/gemma-7b",
        "mistral-22B": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "mistral-7B" : "mistralai/Mistral-7B-Instruct-v0.2",
        "llama-grand-2" : "meta-llama/Meta-Llama-Grand-2-8B",
        "llama-2-13B" : "meta-llama/Llama-2-13b-chat-hf", 
        "llama-2-70B" : "meta-llama/Llama-2-70b-chat-hf", 
        "llama-3-8" : "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama-3-70" : "meta-llama/Meta-Llama-3-70B-Instruct",
    }

def get_model_path(model_name):
    """
    Model names available:
    - gemma = google/gemma-7b
    - mistral-22B = mistralai/Mixtral-8x22B-Instruct-v0.1
    - mistral-7B = mistralai/Mistral-7B-Instruct-v0.2
    - llama-grand-2 = meta-llama/Meta-Llama-Grand-2-8B
    - llama-2-13B = llama-2-13b-chat-hf
    - llama-2-70B = llama-2-70b-chat-hf
    - llama-3-8 = llama-3-8b-chat-hf
    - llama-3-70 = llama-3-70b-chat-hf
    """
    return models[model_name]


def get_model(model_name, 
              quantize,
              access_token):
    # Model
    device_string = PartialState().process_index
    print('Using', device_string)
    model_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_name,
        token=access_token,
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
        device_map= {'': device_string},
        token=access_token
    )

    return model

def get_tokenizer(model_name, access_token):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
        token=access_token
    )
    return tokenizer

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

######################################################################
#############             CODEBOOK FUNCTIONS           ###############
######################################################################

def get_codebook(codebook_file):
    codebook = pd.read_excel(codebook_file)
    codebook[['Class', 'Term']] = codebook['Term'].str.split(":", expand=True)
    codebook['Term'] = codebook['Term'].map(lambda x: str(x.strip()))
    return codebook

def get_formatted_row(row):
    ### Add class name
    message_row = f"### Class: {row['Term']} ###\n"

    ### Add definition
    if not pd.isna(row['Definition']):
        definition = " ".join(row['Definition'].split('\n'))
        message_row += f"### Definition: {definition} ###\n"

    ### Add example
    if not pd.isna(row['Example']):
        example = " ".join(row['Example'].split('\n'))
        message_row += f"### Example: {example} ###\n"

    return message_row

def get_formatted_codebook(codebook, class_to_predict):
    codebook = codebook[codebook['Class'] == class_to_predict].copy()
    codebook.drop(columns=['Class'], inplace=True)
    formatted_codebook = codebook.apply(get_formatted_row, axis=1)
    formatted_codebook = "\n".join(formatted_codebook)
    return formatted_codebook

def get_classes_to_predict(codebook, class_to_predict):
    return codebook[codebook['Class'] == class_to_predict]['Term'].to_list()

def get_classes(codebook):
    return codebook['Class'].unique()


#####################################################################
#############                PREPROCESSING             ##############
#####################################################################

def get_preprocessed_path(model_type, class_to_predict, use_history, use_past_labels, use_context):
    return f'./preprocessed/{model_type}{get_extension(class_to_predict, use_history, use_past_labels, use_context)}'

def get_extension(class_to_predict, use_history, use_past_labels, use_context):
    extension = f'_{class_to_predict.lower()}'
    if use_history or use_past_labels or use_context:
        extension += '_w'
        extension += '_history' if use_history else ''
        extension += '_past-labels' if use_past_labels else ''
        extension += '_context' if use_context else ''
    return extension

def preprocess_data(
        dataset_file,
        combine_fields = [],
        separator = ': ',
        text_field = 'message',
        class_field = '',
        history_field = 'past_chat',
        history_label = 'past_labels',
        unique_keys_for_conversation =  ['book_id', 'bookclub', 'course'],
        window_size = 3,
        use_past_labels = False
):
    data = pd.read_csv(dataset_file)
    if len(combine_fields) > 0:
        data[text_field] = data[combine_fields].apply(lambda x: separator.join(x.dropna().astype(str)), axis=1)

    past_chat = []
    data[history_field] = pd.Series()
    if use_past_labels:
        past_labels = []
        data[history_label] = pd.Series()

    for i in range(len(data)):
        if i >= 1 and not data.iloc[i][unique_keys_for_conversation].equals(data.iloc[i-1][unique_keys_for_conversation]):
            past_chat = []
            if use_past_labels:
                past_labels = []

        data.at[i, history_field] = past_chat.copy()
        past_chat.append(data.iloc[i][text_field])
        
        if use_past_labels:
            data.at[i, history_label] = past_labels.copy()
            past_labels.append(data.iloc[i][class_field])

        if window_size > 0 and len(past_chat) > window_size:
            past_chat.pop(0)
            if use_past_labels:
                past_labels.pop(0)

    return data

def load_data(model_class, class_to_predict, use_history, use_past_labels, use_context):
    final_path = f'./preprocessed/{model_class}{get_extension(class_to_predict, use_history, use_past_labels, use_context)}'
    dataset = load_from_disk(final_path)
    return dataset

def split_data(data, test_size=0.2, random_state=42):
    train_val_dataset = data.train_test_split(test_size=test_size, seed=random_state)
    data['train'] = train_val_dataset['train']
    data['validation'] = train_val_dataset['test']
    return data


#####################################################################
#############                POSTPROCESS              ###############
#####################################################################

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

def preprocess(text):
    tokens = []
    for token in word_tokenize(text.lower()):
        if token.isalpha():# and token not in stop_words:
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
    # add None
    items = np.append(items, 'None')
    for i in items:
        index = sentence.find(preprocess(i))
        if index != -1 and index < first:
            first = index
            item = i
    if item is None:
        item = 'None'
    return item

def get_results_path(model_type, class_to_predict, use_history, use_past_labels, use_context):
    return f'./pred_results/{model_type}{get_extension(class_to_predict, use_history, use_past_labels, use_context)}'

#####################################################################
#############                RETRIEVER                ###############
#####################################################################

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
    def __init__(self, 
                 xmls = [], 
                 websites = [],
                 documents_to_retrieve = 1, 
                 embed_model = "sentence-transformers/all-mpnet-base-v2",
                 use_llm = False,
                 llm_model = "meta-llama/Meta-Llama-3-8B-Instruct",
                 access_token = get_access_token(),
                 quantize = True,
                 ):
        # Loading documents
        docs = []
        for xml_path in xmls:
            docs += parse_xml(xml_path)
        docs += fetch_websites(websites)

        # Splitting documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
        story_splitted = text_splitter.split_documents(docs)

        # Creating retriever
        embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": "cuda"},
        )
        vector_store = FAISS.from_documents(story_splitted, embeddings)
        self.retreiver = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': documents_to_retrieve}
        )

        ### Initializing model, tokenizer and LLM
        self.use_llm = use_llm
        if use_llm:
            # Model
            model_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=llm_model,
                token=access_token,
            )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, # loading in 4 bit
                bnb_4bit_quant_type="nf4", # quantization type
                bnb_4bit_use_double_quant=True, # nested quantization
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=llm_model,
                config=model_config,
                quantization_config=bnb_config if quantize else None,
                device_map= "auto",
                token=access_token
            )
            model.config.use_cache = False
            model.eval()

            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=llm_model,
                trust_remote_code=True,
                token=access_token
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"

            # LLM
            self.pipe = pipeline(
                task="text-generation",
                model=model,
                tokenizer=self.tokenizer,
                return_full_text = False,
                temperature = 0.5,
                max_new_tokens = 500,
                repetition_penalty = 1.5,
                do_sample = True,
            )

            self.system_chat = [
                {
                    "role": "system",
                    "content": "You are an AI chat summarizer. You will be given a conversation history and a new input. Your task is to summarize the conversation and the new input trying to get the essence and topic of an ipotetic story they are talking about. Only summarize this conversation, this is all the content that you will receive. If you fail to summarize, return the same conversation that you have receive as input.",
                },
            ]
        
    def invoke(self, request, history_field = 'chat_history', format_as_user_chat = True):
        if format_as_user_chat:
            query = self.system_chat
            if history_field in request and len(request[history_field]) > 0:
                for user_message in request[history_field]:
                    query.append({"role": "user", "content": user_message})
            query.append({"role": "user", "content": request['input']})
        else:
            query = ''
            if history_field in request and len(request[history_field]) > 0:
                for user_message in request[history_field]:
                    query += user_message + '\n'
            query += request['input']

        if self.use_llm:
            query = self.pipe(query, return_full_text = False)[0]['generated_text']
        
        return self.retreiver.invoke(query)

    def batch_invoke(self, prompt_requests, history_field = 'chat_history', format_as_user_chat = True):
        if format_as_user_chat:
            queries = []
            for request in prompt_requests:
                query = self.system_chat.copy()
                if history_field in request and len(request[history_field]) > 0:
                    for user_message in request[history_field]:
                        query.append({"role": "user", "content": user_message})
                query.append({"role": "user", "content": request['input']})
                queries.append(query)
        else:
            queries = []
            for request in prompt_requests:
                query = ''
                if history_field in request and len(request[history_field]) > 0:
                    for user_message in request[history_field]:
                        query += user_message + '\n'
                query += request['input']
                queries.append(query)

        if self.use_llm:
            if not format_as_user_chat:
                queries = [self.system_chat['content'] + query for query in queries]

            generated = self.pipe(queries, return_full_text = False)
            queries = [query[0]['generated_text'] for query in generated]
            
        return self.retreiver.batch(queries)
    
