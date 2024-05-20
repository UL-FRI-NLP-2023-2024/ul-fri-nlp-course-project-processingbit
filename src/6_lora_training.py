import torch
from transformers import pipeline, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from utils import *

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    model_name = "mistral-7B"
    llm_model = get_model_path(model_name)
    print(f'Model: {llm_model}')
    quantize = True

    # Load data
    class_to_predict = 'Discussion'
    use_history = True
    use_past_labels = True
    use_context = True

    dataset = load_data(model_name, class_to_predict, use_history, use_past_labels, use_context)
    classes = np.unique(dataset['train']['labels'])
    print('Train classes:', classes)
    print('Validation classes:', np.unique(dataset['validation']['labels']))
    print('Test classes:', np.unique(dataset['test']['labels']))

    # Model and tokenizer
    model = get_model(llm_model, quantize)
    tokenizer = get_tokenizer(llm_model)

    ### SETTING TOKENIZER ###
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    #tokenizer.add_bos_token = False
    #tokenizer.add_eos_token = False
    #print("Add bos, eos tokens", tokenizer.add_eos_token, tokenizer.add_bos_token)
    print('eos token:', tokenizer.eos_token)
    test_message = [
        {"role": "user", "content": "SOMETHING"},
        {"role": "assistant", "content": "SOMETHING"}
    ]

    assistant_token = tokenizer.apply_chat_template(test_message, tokenize=False, add_generation_prompt=True)
    pre_assistant_token = assistant_token.split('SOMETHING')[1]
    post_assistant_token = assistant_token.split('SOMETHING')[2]
    print('Pre-assistant token:', pre_assistant_token)
    print('Post-assistant token:', post_assistant_token)

    # Format the text
    def tokenize(element):
        message = element['text'].copy()
        message.append({
            "role": "assistant",
            "content": element['labels']
        })
        tokenized = tokenizer.apply_chat_template(message, tokenize = False)
        output = tokenizer(tokenized, padding=True, max_length=2048)
        return {
            "input_ids": output['input_ids'],
            "attention_mask": output['attention_mask']
        }

    for split in dataset.keys():
        dataset[split] = dataset[split].map(tokenize)
        
    max_token_length = max([len(element['input_ids']) for element in dataset['train']])
    
    ########################## TRAINING ##########################
    
    #model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    #model.config.pad_token_id = model.config.eos_token_id
    #model.config.pretraining_tp = 1
    if quantize:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
    )

    lora_model = get_peft_model(model, peft_config)
    print_trainable_parameters(lora_model)   
    
    training_arguments = TrainingArguments(
        output_dir= f"./checkpoints_{model_name}",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        optim="paged_adamw_32bit",
        num_train_epochs=100,
        logging_steps=20,
        save_steps=20,
        learning_rate=2e-4,
        warmup_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        evaluation_strategy="steps",
    )

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    # NOT WORKING
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
        
        decoded_preds = [find_first(pred, classes) for pred in decoded_preds]

        return {
            'accuracy': accuracy_score(decoded_labels, decoded_preds),
            'precision': precision_score(decoded_labels, decoded_preds, average='weighted'),
            'recall': recall_score(decoded_labels, decoded_preds, average='weighted'),
            'f1': f1_score(decoded_labels, decoded_preds, average='weighted')
        }

    response_template_ids = tokenizer.encode(pre_assistant_token, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=lora_model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=collator,
        args=training_arguments,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Evaluation
    eval_results = trainer.evaluate()
    print(eval_results)
    
    # Training
    trainer.train()

    test_results = trainer.predict(dataset['test'])
    print(test_results)

    # Save the model
    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(f"./adapters/{model_name}{get_extension(class_to_predict, use_history, use_past_labels, use_context)}")

    # Testing
    model.config.use_cache = True
    #tokenizer.add_bos_token = True
    #tokenizer.add_eos_token = False
    #tokenizer.padding_side = 'left'
    model.eval()

    pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            #temperature=0.2,
            max_new_tokens=100,
            repetition_penalty=1.5,
            #do_sample=True,
        )

    generated_texts = pipe(dataset['test']['text'])

    answers = [answer[0]['generated_text'] for answer in generated_texts]

    post_processed_answers = [find_first(answer, classes) for answer in answers]

    for i, class_answer in enumerate(post_processed_answers):
        print('Generated:', answers[i])
        print('Post-processed:', class_answer)
        print('Real Label:', dataset['test']['labels'][i])
        print()

    # Classification report
    print(classification_report(dataset['test']['labels'], answers))

    # Save the generated text
    filepath = get_results_path(model_name, class_to_predict, use_history, use_past_labels, use_context)
    np.save(filepath, post_processed_answers)
    

