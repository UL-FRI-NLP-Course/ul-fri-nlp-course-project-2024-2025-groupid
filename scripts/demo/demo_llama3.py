from transformers import pipeline
import torch
from huggingface_hub import login, delete_repo, scan_cache_dir
import os
import time

login(token=os.getenv("HF_TOKEN"))
# Check GPU availability
print(torch.cuda.is_available())   # Should return True
print(torch.cuda.get_device_name(0))  # Optional: shows your GPU name

# Use the Llama 3 model ID from Hugging Face
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Create the text generation pipeline
pline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Your test prompts
messages = [
    {"role": "user", "content": """Generate a traffic report in slovenian language that:
- Omits all non-data phrases ("be careful", "please note")
- Uses only the two allowed sentence structures:
  a) "Na [cesta] proti [smer] je zaradi [razlog] [posledica] med [lokacija1] in [lokacija2]."
  b) "Zaradi [razlog] je na [cesta] proti [smer] [posledica] v odseku [lokacija]."
- Do not repeat the information. 
     
For border waits, simply state: "Na mejnem prehodu [X] je čakalna doba."

PODATKI: Previdno vozite po štajerski avtocesti od Šempetra proti Žalcu torej proti Mariboru. Na vozišču je voznik, ki vozi v napačno smer. 
         Zastoj je na Bledu. Zastoj je na cesti Bled - Lesce. 
         Zaradi vozila v okvari je zaprt en pas na cesti Črnuče - Tomačevo, proti krožišču Tomačevo. 
         Čakalne dobe pri vstopu: Slovenska vas, Obrežje, Dobovec in Gruškovje. Čakalne dobe pri vstopu: Metlika, Slovenska vas, Obrežje in Dobovec. 
REPORT:
"""},
]

start = time.time()
# Generate sequences
outputs = pline(
    messages, max_new_tokens=256
)
end = time.time()
print(f"Elapsed time {end - start}")

# Extract just the final report
print(outputs[0]["generated_text"][-1])
# Print results
# for seq in sequences:
#     print("--------------------------")
#     print(f"Result: {seq[0]['generated_text']}")
#     print("--------------------------\n")