import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig


# === CONFIGURATION === #
MODEL_ID = "cjvt/GaMS-9B-Instruct"
DATA_PATH = os.path.abspath("/d/hpc/home/ak84795/NLP/data/to_report.json")
OUTPUT_DIR = os.path.abspath("/d/hpc/home/ak84795/NLP/models/combo2_nocot")

# === QUANTIZATION CONFIG === #
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# === LOAD TOKENIZER AND MODEL === #
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    padding_side="right"
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    trust_remote_code=True,
    attn_implementation="eager"
)
model.config.use_cache = False

# === LoRA CONFIG === #
lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model = model.to("cuda")


# === DATA TOKENIZATION === #
def format_json_inputs(json_list):
    return [json.dumps(entry, ensure_ascii=False, indent=2) for entry in json_list]

def tokenize_example(examples):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    base_instruction = (
        "Pretvori vhodne informacije o slovenskem prometnem stanju:\n"
        "- če je vhod v prostem besedilu, ustvari JSON opis;\n"
        "- če je vhod že v JSON formatu, pripravi jasno radijsko poročilo.\n\n"
        #"Vhod je lahko v prostem besedilu ali JSON formatu. Ustvari jasno in natančno radijsko poročilo.\n\n"
        #"če je vhod v prostem besedilu, najprej ustvari JSON opis, ko pa je vhod v JSON formatu, ustvari radijsko poročilo.\n\n"
    )

    for info_text, json_text, reasoning_text, report_text in zip(examples["info"], examples["json"], examples["reasoning"], examples["report"]):

        if info_text:
            # Full prompt (model input)
            full_prompt = (
                base_instruction +
                f"Vhod: {info_text}\n\n"
                #"Razmišljam: Vidim, da je vhod v prostem besedilu. Sedaj bom pretvoril vhod v JSON format.\n\n"
                f"Razmišljam: Vhod je v prostem besedilu. Pretvoril ga bom natančno in dosledno v JSON format.\n\n"
                f"JSON: {json_text}"
                #f"Razmišljam: {reasoning_text} Sedaj bom generiral poročilo.\n\n"
                #f"Poročilo: {report_text}{tokenizer.eos_token}"
            )

            # Prompt prefix (to be masked)
            cutoff_prompt = (
                base_instruction +
                f"Vhod: {info_text}\n\n"
            )

        else:
            full_prompt = (
                base_instruction +
                f"Vhod: {json_text}\n\n"
                #f"Razmišljam: Vhod je v JSON formatu. Preučujem vsebino JSON opisa. {reasoning_text} Zdaj bom pripravil natančno in dosledno poročilo.\n\n"
                f"Razmišljam: Vhod je v JSON formatu. Zdaj bom pripravil natančno in dosledno poročilo.\n\n"
                f"Poročilo: {report_text}{tokenizer.eos_token}"
            )

            cutoff_prompt = (
                base_instruction +
                f"Vhod: {json_text}\n\n"
            )
                

        # Tokenize full input and prefix
        full_tokens = tokenizer(full_prompt, padding=False, truncation=True)
        cutoff_ids = tokenizer(cutoff_prompt, add_special_tokens=False)["input_ids"]
        cutoff_len = len(cutoff_ids)

        # Build labels: mask prefix with -100
        labels = full_tokens["input_ids"].copy()
        labels[:cutoff_len] = [-100] * cutoff_len

        input_ids_list.append(full_tokens["input_ids"])
        attention_mask_list.append(full_tokens["attention_mask"])
        labels_list.append(labels)

    # Pad input_ids and attention_mask
    batch = tokenizer.pad(
        {"input_ids": input_ids_list, "attention_mask": attention_mask_list},
        return_tensors="pt"
    )

    # Manually pad labels to same max length
    max_len = batch["input_ids"].shape[1]
    padded_labels = [
        l + [-100] * (max_len - len(l)) for l in labels_list
    ]
    batch["labels"] = torch.tensor(padded_labels)

    # === DEBUG: Print what the model will actually learn on ===
    for i in range(min(2, len(labels_list))):  # limit to 2 examples per batch
        learnable_tokens = [
            tok for tok, label in zip(input_ids_list[i], labels_list[i]) if label != -100
        ]
        decoded_text = tokenizer.decode(learnable_tokens, skip_special_tokens=True)
        print("\n=== MODEL LEARNS TO PREDICT THIS ===")
        print(decoded_text)
        print("====================================\n")


    return batch


# === LOAD + PREPROCESS DATASET === #
print("Loading dataset...")
raw_dataset = load_dataset("json", data_files=DATA_PATH)

print("Tokenizing dataset...")
tokenized_dataset = raw_dataset.map(tokenize_example, batched=True)

# Remove broken examples (no output tokens)
tokenized_dataset = tokenized_dataset.filter(lambda x: any(tok != -100 for tok in x["labels"]))

# Shuffle and split
split = tokenized_dataset["train"].train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

train_size = len(train_dataset)

# === TRAINING ARGS === #
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=int(train_size/2),
    per_device_eval_batch_size=1,
    num_train_epochs=20,
    weight_decay=0.02,
    # logging_dir="./logs",
    save_steps=20,
    # save_total_limit=2,
    warmup_steps=6,
    fp16=True,
    optim="paged_adamw_32bit",
    lr_scheduler_type="linear",
    label_names=["labels"]
)


# === TRAIN === #
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)
trainer.can_return_loss = True


if __name__ == "__main__":
    print("Starting training...")
    trainer.train()
    print("Training complete.")
