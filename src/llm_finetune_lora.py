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

############################################################
#                   Load the LLM model                     #
############################################################

ACCESS_TOKEN = "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"

LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
CLASS = 'discussion_type'
CLASS_IN_CODEBOOK = 'Discussion'

device = f"cuda:{torch.cuda.current_device()}"

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
    quantization_config=bnb_config, # we introduce the bnb config here.
    device_map="auto"
)

model.config.use_cache = False


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
    return_full_text=True,
    #temperature=0.0,
    max_new_tokens=8192,
    repetition_penalty=1.1,
)

llm = HuggingFacePipeline(pipeline=generate_text)

############################################################
#                           RAG                            #
############################################################

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

embedding = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cuda"},
)

# Template
PROMPT_TEMPLATE = """
You are a helpful AI assistant.
The chat below is between children about a story.
I need you to categorize the new sentence based on the codebook that follows.

### CODEBOOK
{codebook}

Answer with only one of the classes under Term.
You can use the context of the story enclosed by triple backquotes if it is relevant.
If you don't know the answer, just return nan, don't try to make up an answer.

```
{context}
```

### CHAT_HISTORY
{chat_history}

### SENTENCE:
{input}

### ANSWER:
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# Retrive the dataset to test and train the model
dataset = pd.read_csv('cleaned_data/discussion_type.csv')

# Retrieve story from chat history
xml_file = './data/LadyOrThetigerIMapBook.xml'
tree = ET.parse(xml_file)
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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
story_splitted = text_splitter.split_documents(story_doc)
vectorstore = FAISS.from_documents(story_splitted, embedding)
retreiver = vectorstore.as_retriever(
    search_type="similarity",
    k=6,
)

STORY_RETRIEVER_TEMPLATE = """
You are an helpful AI assistant that knows everything about the story in the documents.
You need to retrieve the most relevant document based on the following chat:

CHAT_HISTORY:
{chat_history}

CURRENT_SENTENCE:
{input}
"""

story_retriever_prompt = PromptTemplate.from_template(STORY_RETRIEVER_TEMPLATE)

retriever_chain = create_history_aware_retriever(
    llm=llm,
    retriever=retreiver,
    prompt = story_retriever_prompt,
)

document_chain = create_stuff_documents_chain(llm, prompt_template)

# Retrieve codebook
codebook_file = './data/codebook.xlsx'
codebook = pd.read_excel(codebook_file)
codebook[['Class', 'Term']] = codebook['Term'].str.split(':', expand=True)
codebook = codebook[codebook['Class'] == CLASS_IN_CODEBOOK]
codebook.drop(columns=['Class'], inplace=True)
codebook = codebook.to_string(index=False)

# Construct memory
#memory_buffer = ConversationBufferWindowMemory(
#        memory_key="chat_history", 
#        return_messages=True, 
#        output_key="answer", 
#        llm=llm,
#        k=8,
#    )
#
#def categorize(sentence, codebook, memory):
#    history = memory.load_memory_variables({})
#    chat_history = history['chat_history']
#
#    docs = retriever_chain.invoke({"input": sentence, "chat_history": chat_history})
#
#    response = document_chain.invoke({"input": sentence, "codebook": codebook, "context": docs, "chat_history": chat_history})
#
#    answer = response.split('### ANSWER:\n')[1].strip()
#
#    memory.save_context(inputs={'input': sentence}, outputs={'answer': answer})
#    return answer

# Test the model
#y = dataset[CLASS].values
#dataset.drop(columns=[CLASS], inplace=True)
#answers = []
#for i in range(len(dataset)):
#    if i >= 1 and not dataset.iloc[i][['book_id', 'bookclub', 'course']].equals(dataset.iloc[i-1][['book_id', 'bookclub', 'course']]):
#        memory_buffer.clear()
#
#    sentence = dataset.iloc[i]['message']
#    answer = categorize(sentence, codebook, memory_buffer)
#    print(answer)
#
#    answers.append(answer)
#
#print('Accuracy:', accuracy_score(y, answers))
#print('Precision:', precision_score(y, answers, average='weighted', zero_division=0))
#print('Recall:', recall_score(y, answers, average='weighted', zero_division=0))
#print('F1:', f1_score(y, answers, average='weighted', zero_division=0))
#
#print(classification_report(y, answers, zero_division=0))
    
def categorize_batch(dataset, codebook):
    
    

############################################################
#                           LoRA                           #
############################################################

#from datasets import load_dataset
#
## Load dataset discussion_data.csv
#dataset = load_dataset("csv", data_files="./cleaned_data/discussion_data.csv", split="train")
#
#from peft import LoraConfig, get_peft_model
#from transformers import TrainingArguments
#
## TODO: finetuning
#lora_alpha = 32
#lora_dropout = 0.1
#lora_r = 16
#
#peft_config = LoraConfig(
#    lora_alpha=lora_alpha,
#    lora_dropout=lora_dropout,
#    r=lora_r,
#    bias="none",
#    task_type="CAUSAL_LM"
#)
#
#lora_model = get_peft_model(model, peft_config)
#
## TODO: finetuning
#output_dir = "./llm_results"
#per_device_train_batch_size = 1
#gradient_accumulation_steps = 1
#optim = "paged_adamw_32bit" #specialization of the AdamW optimizer that enables efficient learning in LoRA setting.
#save_steps = 100
#logging_steps = 10
#learning_rate = 2e-4
#max_grad_norm = 0.3
#max_steps = 500
#warmup_ratio = 0.03
#lr_scheduler_type = "constant"
#
#training_arguments = TrainingArguments(
#    output_dir=output_dir,
#    per_device_train_batch_size=per_device_train_batch_size,
#    gradient_accumulation_steps=gradient_accumulation_steps,
#    optim=optim,
#    save_steps=save_steps,
#    logging_steps=logging_steps,
#    learning_rate=learning_rate,
#    fp16=True,
#    max_grad_norm=max_grad_norm,
#    max_steps=max_steps,
#    warmup_ratio=warmup_ratio,
#    group_by_length=True,
#    lr_scheduler_type=lr_scheduler_type,
#    report_to="none"
#)

#from trl import SFTTrainer
#
## TODO: finetuning
#max_seq_length = 512
#
#trainer = SFTTrainer(
#    model=model,
#    train_dataset=dataset,
#    peft_config=peft_config,
#    dataset_text_field="X",
#    max_seq_length=max_seq_length,
#    tokenizer=tokenizer,
#    args=training_arguments,
#)