# Directory made to store test predictions to be loaded from the ensemble

Save the results in numpy files in the format: <method_used><estension>.npy

Estension can be retrieved from the function: 

'''
final_path = f"./preprocessed/data_{get_extension(class_to_predict, use_history, use_past_labels, use_context)}"


def get_extension(class_to_predict, use_history, use_past_labels, use_context):
    extension = f'_{class_to_predict.lower()}'
    if use_history or use_past_labels or use_context:
        extension += '_w'
        extension += '_history' if use_history else ''
        extension += '_past-labels' if use_past_labels else ''
        extension += '_context' if use_context else ''
    return extension
'''