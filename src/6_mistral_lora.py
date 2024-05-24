import torch
from transformers import pipeline, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from datasets import concatenate_datasets

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from utils import *

if __name__ == '__main__':
    ################## SETTINGS ##################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_name = "mistral-7B"
    llm_model = get_model_path(model_name)
    quantize = True
    
    # Settings for the data
    class_to_predict = 'Discussion'
    process_output = True

    use_history = False
    use_past_labels = False
    use_context = True
    extra = None

    # Model and tokenizer
    model = get_model(llm_model, quantize)
    tokenizer = get_tokenizer(llm_model)

    train = False
    train_valid_split = True
    batch_size = 2
    lr = 2e-4
    logging_steps = 20
    warmup_steps = 100
    epochs = 11

    load_adapter = True
    if train_valid_split:
        adapter_file = f"./adapters/{model_name}{get_extension(class_to_predict, use_history, use_past_labels, use_context, extra)}/"
    else:
        adapter_file = f"./adapters/{model_name}{get_extension(class_to_predict, use_history, use_past_labels, use_context, extra)}_final/"
    
    adapter_file = "./checkpoints_mistral-7B_discussion_w_context/checkpoint-200"

    save_output = True
    ########################## SETTING TOKENIZER ######################

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    tokenizer.padding_side = "right"

    test_message = [
        {"role": "user", "content": "SOMETHING"},
        {"role": "assistant", "content": "SOMETHING"}
    ]

    assistant_token = tokenizer.apply_chat_template(test_message, tokenize=False, add_generation_prompt=True)
    pre_assistant_token = assistant_token.split('SOMETHING')[1].strip()
    post_assistant_token = assistant_token.split('SOMETHING')[2].strip()

    if model_name.startswith('llama'):
        pre_assistant_token = '<|start_header_id|>assistant<|end_header_id|>'
    elif model_name.startswith('mistral'):
        pre_assistant_token = '[/INST]'

    ########################## PREPROCESSING ##########################
    def preprocess_element(element):
        message = element['text'].copy()
        message.append({
            "role": "assistant",
            "content": f"Class: {element['labels']}"
        })
        tokenized = tokenizer.apply_chat_template(message, tokenize = False)
        return {'text': tokenized}

    dataset = load_data(model_name, class_to_predict, use_history, use_past_labels, use_context, extra)
    classes = np.unique(dataset['train']['labels'])
    
    dataset['train'] = dataset['train'].map(preprocess_element)
    dataset['validation'] = dataset['validation'].map(preprocess_element)
    dataset['test'] = dataset['test'].map(lambda x: {"text": tokenizer.apply_chat_template(x["text"], tokenize=False)})

    if not train_valid_split:
        dataset['train'] = concatenate_datasets([dataset['train'], dataset['validation']])
    
    ########################## TRAINING ##########################
    print(f'Model: {llm_model}')
    print(f'Class: {class_to_predict}')
    print(f'Training on all data: {not train_valid_split}')
    print("Settings:")
    print(f"- History: {use_history}")
    print(f"- Past labels: {use_past_labels}")
    print(f"- Context: {use_context}")
    print(f"- Extra: {extra}")

    if train:
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        
        if quantize:
            model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.1,
            r=16,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
        print_trainable_parameters(model)   
        
        training_arguments = TrainingArguments(
            output_dir= f"./checkpoints_{model_name}{get_extension(class_to_predict, use_history, use_past_labels, use_context)}",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8,
            optim="paged_adamw_32bit",
            num_train_epochs=epochs,
            logging_steps=logging_steps,
            save_steps=logging_steps if train_valid_split else 0,
            learning_rate=lr,
            warmup_steps=warmup_steps,
            load_best_model_at_end=True if train_valid_split else False,
            metric_for_best_model="f1" if train_valid_split else None,
            evaluation_strategy="steps" if train_valid_split else "no",
        )

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:]
            preds = preds[:, :-1]
            # -100 is a default value for ignore_index used by DataCollatorForCompletionOnlyLM
            mask = labels == -100
            labels[mask] = tokenizer.pad_token_id
            preds[mask] = tokenizer.pad_token_id

            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            
            decoded_preds = [find_first(pred, classes, process_output) for pred in decoded_preds]
            decoded_labels = [find_first(label, classes, process_output) for label in decoded_labels]

            return {
                'accuracy': accuracy_score(decoded_labels, decoded_preds ),
                'precision': precision_score(decoded_labels, decoded_preds, average='weighted', zero_division=0),
                'recall': recall_score(decoded_labels, decoded_preds, average='weighted', zero_division=0),
                'f1': f1_score(decoded_labels, decoded_preds, average='weighted', zero_division=0)
            }

        response_template = pre_assistant_token
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"] if train_valid_split else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            data_collator=collator,
            peft_config=peft_config,
            args=training_arguments,
            compute_metrics=compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=4)] if train_valid_split else None,
            dataset_text_field="text",
            max_seq_length=2048,
            packing=False,
        )

        # Evaluation
        if train_valid_split:
            eval_results = trainer.evaluate()
            print(eval_results)
        
        # Training
        trainer.train()

        # Save the model
        model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(adapter_file)

    elif load_adapter:
        model = PeftModel.from_pretrained(model, model_id = adapter_file)

    # Testing
    model.config.use_cache = True
    model.eval()

    pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            temperature=0.1,
            max_new_tokens=100,
            repetition_penalty=0.9,
            do_sample=True,
        )
    
    generated_texts = pipe(dataset['test']['text'])
    answers = [answer[0]['generated_text'] for answer in generated_texts]

    post_processed_answers = [find_first(answer, classes, process_output) for answer in answers]

    for i, class_answer in enumerate(post_processed_answers):
        print('Generated:', answers[i])
        print('Post-processed:', class_answer)
        print('Real Label:', dataset['test']['labels'][i])
        print()

    # Classification report
    print(classification_report(dataset['test']['labels'], post_processed_answers))

    # Save the generated text
    if save_output:
        filepath = get_results_path(model_name, class_to_predict, use_history, use_past_labels, use_context)
        np.save(filepath, post_processed_answers)
    