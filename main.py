import pandas as pd
import glob
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

#Essa função é responsável por ler e separar o dataset em treinamento, validação e teste
def get_data():
    csv_files = glob.glob('CachacaNER/csv/particao_*.csv')

    df_train = []
    df_test = []
    df_val = []

    partition = 1

    for file in csv_files:
        if(partition <= 7):
            df_train.append(pd.read_csv(file))
        elif(partition == 8):
            df_val.append(pd.read_csv(file))
        else:
            df_test.append(pd.read_csv(file))
        
        partition+=1

    train_dataset = pd.concat(df_train, ignore_index=True)
    val_dataset = pd.concat(df_val, ignore_index=True)
    test_dataset = pd.concat(df_test, ignore_index=True)

    return train_dataset, val_dataset, test_dataset

def main():
    
    train_dataset, val_dataset, test_dataset = get_data()

    base_model = 'maritaca-ai/sabia-7b'
    #base_model = 'berchielli/cabrita-7b-pt-br'

    new_model = 'sabia-2-7b-cachaca-ner'
    #new_model = 'cabrita2-7b-cachaca-ner'

    compute_dtype = getattr(torch, "float16")

    #Configurando parâmetros de quantização
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map={"": 0}
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    #Escolhemos aqui o tipo de task TOKEN_CLS para a classificação de tokens.
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="TOKEN_CLS",
    )

    #Parâmetros de treinamento, por enquanto vou chumbar valores, depois vamos executar com outras combinações
    training_params = TrainingArguments(
        output_dir="./results-fine-tuning",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    #Criando 
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.model.save_pretrained(new_model)
    trainer.tokenizer.save_pretrained(new_model)

    #Verificar os resultados das predições
    predictions = train_result.predict(test_dataset)

    print(predictions)



if __name__ == "__main__":
    main()

