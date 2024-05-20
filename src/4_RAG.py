import pandas as pd
from utils import *

####### FILE CONFIGURATION #######

    
##### MAIN FUNCTION ######

if __name__ == '__main__':

    use_history = True
    quantize = True
    window_size = -1
    dataset_file = "./data/cleaned_data.csv"

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
        dataset_file=dataset_file,
        text_field = text_field,
        history_field = history_field,
        window_size = window_size,
        use_past_labels=False,
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
