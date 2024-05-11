import os
import tempfile
from pathlib import Path
import joblib
import requests
import torch
import transformers

#from IPython.display import print
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores.faiss import FAISS


############################################################
#                   Load the LLM model                     #
############################################################

ACCESS_TOKEN = "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"

LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
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







############################################################
#                           LoRA                           #
############################################################

#from datasets import load_dataset

# Load dataset discussion_data.csv
#dataset = load_dataset("csv", data_files="discussion_data.csv", split="train")
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
#output_dir = "./results"
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
#
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