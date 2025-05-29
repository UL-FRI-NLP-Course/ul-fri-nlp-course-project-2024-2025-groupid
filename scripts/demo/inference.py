from transformers import pipeline
import torch
import os
import json

#model_path = "cjvt/GaMS-9B-Instruct"
model_path = "/d/hpc/home/ak84795/NLP/models/combo2/checkpoint-20"
examples_path = "/d/hpc/home/ak84795/NLP/data/eval.json"

pline = pipeline(
    "text-generation",
    model=model_path,
    device_map={"": 0},
    torch_dtype=torch.float16
)

model = pline.model
print(model.config.max_position_embeddings)
eos_token_id = model.config.eos_token_id

# Load the evaluation.json file
with open(examples_path, "r", encoding="utf-8") as f:
    examples = json.load(f)

# Base instruction prefix
base_instruction = (
    "Pretvori vhodne informacije o slovenskem prometnem stanju:\n"
    "- če je vhod v prostem besedilu, ustvari JSON opis;\n"
    "- če je vhod že v JSON formatu, pripravi jasno radijsko poročilo in pri tem razmišljaj korak za korakom.\n\n"
)


# Create prompts array automatically from evaluation.json inputs
prompts = [base_instruction + "Vhod: " + example["json"] + "\n\n" for example in examples]


sequences = pline(
    prompts,
    max_new_tokens=1000,
    do_sample=False
)

# Print results
for prompt, seq in zip(prompts, sequences):
    full_text = seq[0]['generated_text']
    print("\n--------------------------")
    print("Generated Full Output:\n")
    print(full_text)

    # generated_part = full_text.replace(prompt, "").strip()
    # print("\nExtracted Generation:\n")
    # print(generated_part)
    print("--------------------------\n")
