import tempfile
import requests
import torch
from transformers import AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import xml.etree.ElementTree as ET
import re
import pandas as pd
import numpy as np
from datasets import DatasetDict, load_dataset, Dataset

#from IPython.display import print
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.docstore.document import Document

####### MODEL CONFIGURATION #######
with open("./src/tokens/hugging_face_token.txt", "r") as file:
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
CODEBOOK_FILE = "./src/data/codebook.xlsx"
DATASET_FILE = "./src/data/cleaned_data.csv"

######### PROMPT CONFIGURATION ########

INITIAL_PROMPT = "You are an AI expert in categorizing sentences.\nYou need to categorize the new sentence into one of the following classes: [{classes}].\nIf you fail to categorize the sentence, return 'None' instead of coming up with a wrong class."

INPUT = """
### NEW SENTENCE:
{input}
###
Remember to answer with only the name of the class and nothing else.
"""

CODEBOOK_PROMPT = """You can use the following codebook (with classes, definitions and examples) to help you classify the sentence:
### IMPORTANT CODEBOOK:
{codebook}
###
"""

HISTORY = """You can use the following chat history if it is relevant:
### CHAT HISTORY:
{history}
###
"""

CONTEXT = """Here are some relevant documents that might help you to classify the sentence:
### CONTEXT:
{context}
###
"""




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
    def __init__(self, xmls = [], websites = [], quantize = False, documents_to_retrieve = 3):
        # Loading documents
        docs = []
        for xml_path in xmls:
            docs += parse_xml(xml_path)
        docs += fetch_websites(websites)

        # Splitting documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        story_splitted = text_splitter.split_documents(docs)

        # Creating retriever
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
        vector_store = FAISS.from_documents(story_splitted, embeddings)
        retreiver = vector_store.as_retriever(
            search_type="similarity",
            k=documents_to_retrieve,
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
        model.config.use_cache = False
        model.eval()

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=LLM_MODEL,
            trust_remote_code=True,
            token=ACCESS_TOKEN
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

        # LLM
        llm = HuggingFacePipeline(pipeline=pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text = False,
            temperature = 0.2,
            max_new_tokens = 100,
            repetition_penalty = 1.5,
            do_sample=True,
        ))

        # Retriever prompt
        story_retriever_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an helpful AI assistant that knows everything about the story in the documents.\nYou need to retrieve the most relevant document based on the following chat:"),
            ("placeholder", "{chat_history}"),
            ("human", "{input}")
        ])

        # Final retriever chain
        self.retriever_chain = create_history_aware_retriever(
            llm = llm,
            retriever = retreiver,
            prompt = story_retriever_prompt,
        )

    def invoke(self, request):
        return self.retriever_chain.invoke(request)
        
    def batch(self, requests):
        return self.retriever_chain.batch(requests)
    

##### CODEBOOK FUNCTIONS  ######

def get_codebook():
    codebook = pd.read_excel(CODEBOOK_FILE)
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
        if len(history) > window_size:
            history.pop(0)
    return data

###### RETRIEVAL FUNCTIONS ######

def combine_docs(docs):
    context = '====== DOCUMENT ======\n\n'
    context += '\n\n====== DOCUMENT ======\n\n'.join([doc.page_content for doc in docs])
    return context

##### MAIN FUNCTION ######

if __name__ == '__main__':

    use_history = True
    use_context = False
    quantize = True

    # get codebook
    codebook = get_codebook()
    classes = get_classes(codebook)
    class_to_predict = 'Uptake'
    
    print(f'Processing class: {class_to_predict}')
    # get formatted codebook and classes to predict
    formatted_codebook = get_formatted_codebook(codebook, class_to_predict)
    classes_to_predict = get_classes_to_predict(codebook, class_to_predict)

    # Initial prompt
    final_prompt = INITIAL_PROMPT.format(classes = ', '.join(classes_to_predict))

    # Input prompt
    final_prompt += INPUT

    # Codebook prompt
    final_prompt += CODEBOOK_PROMPT.format(codebook = formatted_codebook)

    # History prompt
    if use_history:
        final_prompt += HISTORY

    # Context prompt
    if use_context:
        retriever = Retriever(xmls = ['./data/LadyOrThetigerIMapBook.xml'], 
                            quantize = quantize,
                            documents_to_retrieve = 3
                            )
        final_prompt += CONTEXT
        
    # Data processing
    text_field = 'message'
    history_field = 'history'

    data = preprocess_data(
        combine_fields = [],
        separator = ': ',
        text_field = text_field,
        history_field = history_field,
        unique_keys_for_conversation =  ['book_id', 'bookclub', 'course'],
        window_size = 3,
    )

    # Prompt requests with input and w/wo history
    if use_history:
        prompt_requests = data.apply(lambda x: {'input': x[text_field],\
                                                'chat_history': [("human", chat) for chat in x[history_field].split('\n')] if not pd.isna(x[history_field]) else [],\
                                                'history': x[history_field]}, axis=1).to_list()
    else:
        prompt_requests = data.apply(lambda x: {'input': x[text_field]}, axis=1).to_list()

    if use_context:
        # Retrieval
        responses = retriever.batch(prompt_requests)

        # Add the retrieved context to the requests
        for i in range(len(prompt_requests)):
            prompt_requests[i]['context'] = combine_docs(responses[i])

    # FINAL PROMPT
    final_prompts = [final_prompt.format(**request) for request in prompt_requests]

    print('Final prompt:')
    print(final_prompts[1])

    # remove \n from the final prompts
    final_prompts = [prompt.replace('\n', ' ') for prompt in final_prompts]

    # COMBINE PROMPTS WITH FINAL CLASSES IN PANDAS
    final_dataset = Dataset.from_dict({
        'prompt': final_prompts,
        'label': data[class_to_predict]
    })

    # Split into train and test
    final_dataset = final_dataset.train_test_split(test_size=0.2, seed=42)
    print('Final dataset:', final_dataset)

    # Save final data
    final_path = f'./preprocessed/dataset_{class_to_predict}'
    final_path += '_with_history' if use_history else ''
    final_path += '_with_context' if use_context else ''

    final_dataset.save_to_disk(final_path)
