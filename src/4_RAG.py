import pandas as pd
from utils import *

####### FILE CONFIGURATION #######
DATASET_FILE = "./data/cleaned_data.csv"
    
##### MAIN FUNCTION ######

if __name__ == '__main__':

    use_history = True
    quantize = True
    window_size = -1

    # Use default llm and retirever
    retriever = Retriever(xmls = ['./data/LadyOrThetigerIMapBook.xml'], 
                            documents_to_retrieve = 5,
                            use_llm=True,
                            quantize = quantize,
                            )
            
    # Data processing
    text_field = 'message'
    history_field = 'history'

    data = preprocess_data(
        dataset_file=DATASET_FILE,
        combine_fields = [],
        separator = ': ',
        text_field = text_field,
        history_field = history_field,
        unique_keys_for_conversation =  ['book_id', 'bookclub', 'course'],
        window_size = window_size,
    )

    # Prompt requests with input and w/wo history
    if use_history:
        prompt_requests = data.apply(lambda x: {'input': x[text_field],
                                                'chat_history': x[history_field],
                                                }, axis=1).to_list()
    else:
        prompt_requests = data.apply(lambda x: {'input': x[text_field]}, axis=1).to_list()

    
    # Retrieval
    docs_data = retriever.batch_invoke(prompt_requests, history_field = 'chat_history')

    dataset = pd.DataFrame(docs_data)

    # Save dataset
    dataset.to_csv(f"./preprocessed/context.csv", index=False)
