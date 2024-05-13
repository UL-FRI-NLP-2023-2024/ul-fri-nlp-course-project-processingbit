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
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter, TokenTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.docstore.document import Document

ACCESS_TOKEN = "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"

LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

CLASS = 'discussion_type'
CLASS_IN_CODEBOOK = 'Discussion'

def initialize_llm():
    # INITIALIZE MODEL
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True, # loading in 4 bit
        bnb_4bit_quant_type="nf4", # quantization type
        bnb_4bit_use_double_quant=True, # nested quantization
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_config = transformers.AutoConfig.from_pretrained(
        pretrained_model_name_or_path=LLM_MODEL,
        token=ACCESS_TOKEN,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=LLM_MODEL,
        config=model_config,
        #quantization_config=bnb_config, # we introduce the bnb config here.
        device_map="auto"
    )
    
    model.config.use_cache = False
    
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=LLM_MODEL,
        trust_remote_code=True,
        token=ACCESS_TOKEN,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    #TODO: finetune the model
    generate_text = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        #temperature=0.0,
        max_new_tokens=8192,
        repetition_penalty=1.1,
    )

    llm = HuggingFacePipeline(pipeline=generate_text)
    return llm

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

start = """
You are a helpful AI classifier that knows everything about discourse analysis and can help children to classify the discussion they are having.
"""

codebook_template = """
You have to classify the NEW_SENTENCE with one term from the codebook here:
***
{codebook}
***

If you failed to classify the sentence, return None.
"""

context = """
You can use the following context, if it is relevant.
'''
{context}
'''
"""

history = """
The children are discussing a topic. Here is the chat history:
[
{chat_history}
]
"""

input = """
What class does the following sentence belong to?

### NEW_SENTENCE: "{input}"

Return the class from the codebook or None if you can't classify it.
Return the class name enclosed in **double asterisks**.

### ANSWER:"""

class LLM:
    def __init__(self, 
                 codebook_file,
                 use_xmls = [],
                 use_websites = [],
                 use_custom_history = True,
                 use_buffer = False,
                 ):
        self.llm = initialize_llm()
        if len(use_xmls) != 0 or len(use_websites) != 0:
            self.retriever_chain = self.get_retriever_chain(xml=use_xmls, web=use_websites)
        else:
            self.retriever_chain = None

        self.codebook = self.get_codebook(codebook_file)

        if use_buffer:
            self.memory_buffer = ConversationBufferWindowMemory(
                memory_key="chat_history", 
                return_messages=True, 
                output_key="answer", 
                llm=self.llm,
                k=8,
            )
        else:
            self.memory_buffer = None
        
        self.use_custom_history = use_custom_history

        self.llm_chain = self.get_chain()

    def get_retriever_chain(self, xml = [], 
                        web = [],
                        template =
                            """
                            You are an helpful AI assistant that knows everything about the story in the documents.
                            You need to retrieve the most relevant document based on the following chat:

                            CHAT_HISTORY:
                            {chat_history}

                            CURRENT_SENTENCE:
                            {input}
                            """):
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

        story_retriever_prompt = PromptTemplate.from_template(template)

        retriever_chain = create_history_aware_retriever(
            llm=self.llm,
            retriever=retreiver,
            prompt = story_retriever_prompt,
        )
        return retriever_chain
    
    def get_codebook(self, codebook_file):
        codebook = pd.read_excel(codebook_file)
        codebook[['Class', 'Term']] = codebook['Term'].str.split(':', expand=True)
        codebook = codebook[codebook['Class'] == CLASS_IN_CODEBOOK]
        codebook.drop(columns=['Class'], inplace=True)
        return codebook.to_string(index=False)
        
    def clear_memory(self):
        if self.memory_buffer:
            self.memory_buffer.clear()

    def get_template(self):
        final_template = start

        if self.retriever_chain is not None:
            final_template += context
    
        if self.codebook is None:
            raise Exception("Codebook is not initialized")
        
        final_template += codebook_template.format(codebook=self.codebook)
        
        if self.use_custom_history or self.memory_buffer is not None:
            final_template += history
        
        final_template += input

        return ChatPromptTemplate.from_template(final_template)
    
    def get_chain(self):
        if self.retriever_chain is not None:
            chain = create_stuff_documents_chain(self.llm, self.get_template())
        else:
            chain = self.get_template() | self.llm
        
        return chain

    def single_predict(self, sentence, chat_history = None):
        if self.llm_chain is None:
            raise Exception("Chain is not initialized")
        
        request = {}
        request['input'] = sentence
        
        if self.memory_buffer is not None and chat_history is None:
            history = self.memory_buffer.load_memory_variables({})
            request['chat_history'] = history['chat_history']

        if self.use_custom_history:
            if chat_history is None:
                request['chat_history'] = ''
            else:
                request['chat_history'] = chat_history
        
        if self.retriever_chain is not None:
            docs = self.retriever_chain.invoke(request)
            request['context'] = docs
        
        response = self.llm_chain.invoke(request)
        print(response)

        answer = response.split('**')[1].strip()

        if self.memory_buffer is not None:
            self.memory_buffer.save_context(inputs={'input': sentence}, outputs={'answer': answer})
        
        return answer

    def predict_batch(self, sentences, chat_histories = None):
        if self.llm_chain is None:
            raise Exception("Chain is not initialized")
        
        requests = []
        for i in range(len(sentences)):
            request = {}
            request['input'] = sentences[i]

            if self.use_custom_history:
                if chat_histories is None:
                    request['chat_history'] = ''
                else:
                    request['chat_history'] = chat_histories[i]
            requests.append(request)

        if self.retriever_chain is not None:
            list_docs = self.retriever_chain.batch(requests)
            for i in range(len(requests)):
                requests[i]['context'] = list_docs[i]
        
        responses = self.llm_chain.batch(requests)
        return responses



my_llm_classifier = LLM(codebook_file = './data/codebook.xlsx',
                        use_xmls = [], #['./data/LadyOrThetigerIMapBook.xml'],
                        use_websites = [],
                        use_custom_history = False,
                        use_buffer = False
                        )

# Test the model
# Retrive the dataset to test and train the model
dataset = pd.read_csv('cleaned_data/discussion_type.csv')
y = dataset[CLASS].values
dataset.drop(columns=[CLASS], inplace=True)
answers = []
for i in range(len(dataset)):
    if i >= 1 and not dataset.iloc[i][['book_id', 'bookclub', 'course']].equals(dataset.iloc[i-1][['book_id', 'bookclub', 'course']]):
        memory_buffer.clear()

    sentence = dataset.iloc[i]['message']
    answer = categorize(sentence, codebook, memory_buffer)
    print(answer)

    answers.append(answer)

print('Accuracy:', accuracy_score(y, answers))
print('Precision:', precision_score(y, answers, average='weighted', zero_division=0))
print('Recall:', recall_score(y, answers, average='weighted', zero_division=0))
print('F1:', f1_score(y, answers, average='weighted', zero_division=0))

print(classification_report(y, answers, zero_division=0))
    