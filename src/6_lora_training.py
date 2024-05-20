import torch
from transformers import pipeline, TrainingArguments
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
    
    model_name = "llama-3-8"
    llm_model = get_model_path(model_name)
    print(f'Model: {llm_model}')

    use_adapter = True
    load_adapter = False
    quantize = True

    # Load data
    class_to_predict = 'Discussion'
    use_history = True
    use_past_labels = True
    use_context = True

    dataset = load_data(model_name, class_to_predict, use_history, use_past_labels, use_context)
    dataset = split_data(dataset, test_size=0.2, random_state=42)
    classes = np.unique(dataset['train']['labels'])

    # Model and tokenizer
    model = get_model(llm_model, quantize)
    tokenizer = get_tokenizer(llm_model)

    ### SETTING TOKENIZER ###
    tokenizer.pad_token = tokenizer.eos_token
    print("EOS token/id", tokenizer.eos_token, tokenizer.get_vocab()[tokenizer.eos_token])
    tokenizer.padding_side = "right"
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    print("Add bos, eos tokens", tokenizer.add_eos_token, tokenizer.add_bos_token)
    print('Assistant token:', tokenizer.apply_chat_template("", tokenize=False, add_generation_prompt=True))

    # Format the text
    dataset = dataset.map(lambda x: {"text": tokenizer.apply_chat_template(x["text"], tokenize=False, add_generation_prompt=True)})
    
    # max token length
    tokens = tokenizer(dataset['train']['text'], padding=True, return_tensors='pt')
    max_tokens = tokens['input_ids'].shape[1]
    print('Max tokens:', max_tokens)
    
    ########################## TRAINING ##########################
    
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
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
        num_train_epochs=20,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        #gradient_checkpointing_kwargs={'use_reentrant': False},
        optim="paged_adamw_32bit",

        save_strategy='steps',
        save_steps=30,

        logging_steps=30,
        evaluation_strategy='steps',
        #eval_acculamtion_steps=1,
        #eval_steps=20,

        learning_rate=2e-4,
        #fp16=True,
        #max_grad_norm=0.3,
        #max_steps=-1,
        #group_by_length=True,
        lr_scheduler_type="linear", # "linear", "cosine"
        warmup_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    # Setting sft parameters
    def formatting_prompts_func(example):
        output_texts = []
        for formatted_text, label in zip(example['text'], example['labels']):
            text = f"{formatted_text} {label}"
            output_texts.append(text)
        return output_texts

    def compute_metrics(pred):
        #label_ids = pred.label_ids
        #pred_ids = pred.predictions.argmax(-1)
        #pred_ids[pred_ids == -100] = tokenizer.pad_token_id
        #pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        #label_ids[label_ids == -100] = tokenizer.pad_token_id
        #label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        predictions, labels = pred.predictions, pred.label_ids
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels[labels == -100] = tokenizer.pad_token_id
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        preds = [find_first(text, classes) for text in predictions]
        
        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='weighted'),
            'recall': recall_score(labels, preds, average='weighted'),
            'f1': f1_score(labels, preds, average='weighted')
        }

    response_template = tokenizer.apply_chat_template("", tokenize=False, add_generation_prompt=True)
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        peft_config=peft_config,
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Training
    trainer.train()

    # Evaluation
    eval_results = trainer.evaluate()
    print(eval_results)

    test_results = trainer.predict(dataset['test'])
    print(test_results)

    # Save the model
    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(f"./adapters/{model_name}{get_extension(class_to_predict, use_history, use_past_labels, use_context)}")

    # Testing
    model.config.use_cache = True
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = False
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
    
