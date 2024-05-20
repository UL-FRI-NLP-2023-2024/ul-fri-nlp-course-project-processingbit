import torch
from transformers import pipeline, TrainingArguments
from transformers import EarlyStoppingCallback

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from utils import *

model_name = "mistral-7B"
model_class = "mistral"
LLM_MODEL = get_model_path(model_name)
print(f'Model: {LLM_MODEL}')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load data
    class_to_predict = 'Discussion'
    use_history = True
    use_past_labels = True
    use_context = False
    num_docs = 2

    dataset = get_data_for_train_test(
        class_to_predict=class_to_predict,
        use_history=use_history,
        use_past_labels=use_past_labels,
        num_docs=num_docs,
        model_name=model_class)

    dataset = split_data(dataset, test_size=0.2, random_state=42)
    classes = np.unique(dataset['train']['labels'])

    # Model and tokenizer
    use_adapter = True
    load_adapter = False
    quantize = True

    model = get_model(LLM_MODEL, quantize)
    tokenizer = get_tokenizer(LLM_MODEL)

    ### SETTING TOKENIZER ###
    tokenizer.pad_token = tokenizer.eos_token
    print("EOS token/id", tokenizer.eos_token, tokenizer.get_vocab()[tokenizer.eos_token])
    tokenizer.padding_side = "right"
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    print("Add bos, eos tokens", tokenizer.add_eos_token, tokenizer.add_bos_token)
    
    #pad_token = "[PAD]"     
    #tokenizer.add_special_tokens({'pad_token': pad_token})
    #tokenizer.padding_side = 'right'

    #print("PAD token/id", pad_token, tokenizer.convert_tokens_to_ids(pad_token))

    dataset = dataset.map(lambda x: {"formatted_text": tokenizer.apply_chat_template(x["text"], tokenize=False, add_generation_prompt=False)})
    print("Formatted sentence: ", dataset['train']['formatted_text'][0])
    print()
    # tokenize the first example
    example_token = tokenizer(dataset['train']['formatted_text'][0], padding=True, return_tensors='pt')
    example_sentence = tokenizer.decode(example_token['input_ids'][0])
    print('Example sentence: ', example_sentence)
    print()

    print('Assistant token:', tokenizer.apply_chat_template("", tokenize=False, add_generation_prompt=True))

    # max token length
    tokens = tokenizer(dataset['train']['formatted_text'], padding=True, return_tensors='pt')
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
        output_dir="./checkpoints",
        num_train_epochs=10,
        
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        #auto_find_batch_size=True, # FIND THE BEST BATCH SIZE THAT FEEDS THE GPU

        gradient_accumulation_steps=4,
        #gradient_checkpointing_kwargs={'use_reentrant': False},
        optim="paged_adamw_32bit",

        save_strategy='steps',
        save_steps=200,

        evaluation_strategy='steps',
        eval_steps=50,

        learning_rate=2e-4,
        #fp16=True,
        #max_grad_norm=0.3,
        #max_steps=-1,
        #group_by_length=True,
        lr_scheduler_type="linear", # "linear", "cosine"
        warmup_ratio=0.1,
        report_to="none",

        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    # Setting sft parameters
    def formatting_prompts_func(example):
        output_texts = []
        for formatted_text, label in zip(example['formatted_text'], example['labels']):
            text = f"{formatted_text} {label}"
            output_texts.append(text)
        return output_texts

    def compute_metrics(pred):
        label_ids = pred.label_ids
        pred_ids = pred.predictions.argmax(-1)

        pred_ids[pred_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        print('Generated:', pred_str)

        preds = [find_first(text, classes) for text in pred_str]
        labels = label_str

        print('Preds:', preds)
        print('Labels:', labels)

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
    
    # Hyperparameter search
    #best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")
    #print(best_run)
    #for n, v in best_run.hyperparameters.items():
    #    setattr(trainer.args, n, v)
    
    # Training
    trainer.train()

    # Evaluation
    #eval_results = trainer.evaluate()
    #print(eval_results)

    # Save the model
    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(f"./adapters/mistral{get_extension(class_to_predict, use_history, use_context)}")

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

    generated_texts = pipe(dataset['test']['formatted_text'])

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
    filepath = get_results_path(model_class, class_to_predict, use_history, use_past_labels, use_context)
    np.save(filepath, post_processed_answers)
    

