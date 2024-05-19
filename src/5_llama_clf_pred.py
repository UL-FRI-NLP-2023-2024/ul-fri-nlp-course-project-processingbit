from transformers import AutoModelForSequenceClassification
import torch
from transformers import AutoConfig, BitsAndBytesConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import TextClassificationPipeline

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from accelerate import PartialState

from datasets import load_from_disk
from peft import LoraConfig, PeftConfig, get_peft_model, PeftModel

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

LLM_MODEL = models["llama-3-8"]
print(f'Model: {LLM_MODEL}')

quantize = True
dataset_file = './preprocessed/llama_discussion_w_history_past-labels'
text_field = 'text'

############# DATASET FOR TRAINING AND TEST ################
data_with_test = load_from_disk(dataset_file)
data = data_with_test['train']
test_data = data_with_test['test']

####### MODEL ##################
# INITIALIZE MODEL
device_string = PartialState().process_index

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # loading in 4 bit
    bnb_4bit_quant_type="nf4", # quantization type
    bnb_4bit_use_double_quant=True, # nested quantization
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def get_extension(class_to_predict, use_history, use_past_labels, use_context):
    extension = f'_{class_to_predict.lower()}'
    if use_history or use_past_labels or use_context:
        extension += '_w'
        extension += '_history' if use_history else ''
        extension += '_past-labels' if use_past_labels else ''
        extension += '_context' if use_context else ''
    return extension


final_path = f"./pred_results/llamaForClassification_{get_extension('Discussion', True, True, False)}"

# Fakes fix it 
classes_to_predict = np.unique(data['labels'])
id2label = {i: label for i, label in enumerate(classes_to_predict)}
label2id = {label: i for i, label in enumerate(classes_to_predict)}

model_config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=LLM_MODEL,
    token=ACCESS_TOKEN,
    num_labels = len(classes_to_predict),
)

model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=LLM_MODEL,
    config=model_config,
    quantization_config=bnb_config if quantize else None,
    device_map={'':device_string},
    token=ACCESS_TOKEN,
)

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=LLM_MODEL,
    trust_remote_code=True,
    token=ACCESS_TOKEN
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

test_data = test_data.map(lambda x: {"text": tokenizer.apply_chat_template(x["text"], tokenize=False, add_generation_prompt=True).replace(tokenizer.eos_token, "[eos]")})

model.config.use_cache = False

model = PeftModel.from_pretrained(model, model_id = './checkpoints/checkpoint-280', peft_config = bnb_config)
#model = PeftModel.from_pretrained(model, model_id = './clf-new_format_fixed_history_lr_2e4_128', peft_config = bnb_config)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False)

out = pipe(test_data['text'])

preds = []
for el in out:
    num = el['label'][-1]
    idx = int(num)
    preds.append(id2label[idx])

labels = test_data['labels']

accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds, average='weighted')
recall = recall_score(labels, preds, average='weighted')
f1 = f1_score(labels, preds, average='weighted')

print(f"Classification Report:\n{classification_report(labels, preds)}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")