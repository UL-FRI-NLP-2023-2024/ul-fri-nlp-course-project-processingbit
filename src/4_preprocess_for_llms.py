import pandas as pd
import numpy as np
from datasets import Dataset

from utils import *

####### MODEL CONFIGURATION #######
model_name = "mistral-7B"
model_type = "mistral"
LLM_MODEL = get_model_path(model_name)
ACCESS_TOKEN = get_access_token()

####### FILE CONFIGURATION #######
CODEBOOK_FILE = "./data/codebook.xlsx"
DATASET_FILE = "./data/cleaned_data.csv"

######### PROMPT CONFIGURATION ########

INITIAL_PROMPT = """You are an AI expert in categorizing sentences into classes."""

CONTEXT = "Another assistant has retrieved some documents that might be useful for you to understand the context of the conversation, do not use them if not relevant."

CODEBOOK_PROMPT = """You can use the following codebook (with classes, definitions and examples) to help you ccategorize the sentence:
### IMPORTANT CODEBOOK:
{codebook}
###
You need to categorize the new sentence into one of the following classes: [{classes}].
If you fail to categorize the sentence, return 'None' instead of coming up with a wrong class.
"""

HISTORY = "The following is the history of the conversation:"

##### MAIN FUNCTION ######

if __name__ == '__main__':

    use_history = True
    use_past_labels = True
    num_docs_as_context = 0

    class_to_predict = 'Discussion'

    model2id = {
        "mistral" : 0,
        "llama" : 1
    }
    id2model = {v: k for k, v in model2id.items()}
    model_used = model2id["mistral"]

    # get codebook
    codebook = get_codebook(CODEBOOK_FILE)
    classes = get_classes(codebook)
    
    print(f'Processing class: {class_to_predict}')
    # get formatted codebook and classes to predict
    formatted_codebook = get_formatted_codebook(codebook, class_to_predict)
    classes_to_predict = get_classes_to_predict(codebook, class_to_predict)

    # messages
    first_message = INITIAL_PROMPT
    codebook_message = CODEBOOK_PROMPT.format(codebook = formatted_codebook, classes = ", ".join(classes_to_predict))
        
    # Data processing
    text_field = 'message'
    history_field = 'past_chat'
    history_labels = 'past_labels'

    data = preprocess_data(
        combine_fields = [],
        separator = ': ',
        text_field = text_field,
        class_field= class_to_predict,
        history_field = history_field,
        history_label = history_labels,
        unique_keys_for_conversation =  ['book_id', 'bookclub', 'course'],
        window_size = 6,
        use_past_labels = use_past_labels
    )

    # getting the context
    if num_docs_as_context > 0:
        context = pd.read_csv("./preprocessed/context.csv")

    # Add input
    prompts = []
    for ind, row in data.iterrows():
        system_message = first_message

        if num_docs_as_context > 0:
            system_message += f"\n{CONTEXT}"
            docs = context.iloc[ind].to_list()
            for doc in docs[:num_docs_as_context]:
                system_message += f"\n{doc}"

        system_message += f"\n{codebook_message}"

        if use_history:
            system_message += f"\n{HISTORY}"

        # No system for mistral
        if model_used == model2id["mistral"]:
            message = [{ "role": "user", "content": system_message}]
            message.append({ "role": "assistant", "content": "Ok, let's start!"})

        # System for llama
        if model_used == model2id["llama"]:
            message = [{ "role": "system", "content": system_message}]
        
        if use_history:
            for i, chat in enumerate(row[history_field]):
                message.append({ "role": "user", "content": chat})
                if use_past_labels:
                    message.append({ "role": "assistant", "content": f'Class: {row[history_labels][i]}'})
        
        message.append({ "role": "user", "content": row[text_field]})
        prompts.append(message)

    # COMBINE PROMPTS WITH FINAL CLASSES IN PANDAS
    labels = [str(label) for label in data[class_to_predict]]
    indexes = data.index

    final_dataset = Dataset.from_dict({
        'index': indexes,
        'text': prompts,
        'labels': labels
    })

    # Split into train and test
    final_dataset = final_dataset.train_test_split(test_size=0.2, seed=42)
    print('Final dataset:', final_dataset)

    # Save final data
    final_path = get_preprocessed_path(model_type, class_to_predict, use_history, use_past_labels, num_docs_as_context > 0)

    final_dataset.save_to_disk(final_path)
