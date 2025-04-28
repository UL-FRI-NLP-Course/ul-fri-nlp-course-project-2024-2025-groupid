from transformers import pipeline
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

model_id = "cjvt/GaMS-9B"

pline = pipeline(
    "text-generation",
    model=model_id,
    device_map={"": 0},
    torch_dtype=torch.float16
)

prompts = ["""Generate a traffic report in slovenian language that:
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
"""]

sequences = pline(
    prompts,
    max_new_tokens=1000,      # Control the length more precisely
    do_sample=False,
    # temperature=0.8,         # Slightly lower for more coherence
    # top_k=5,                # Allow more token options
    # top_p=0.99,              # Add nucleus sampling
    repetition_penalty=1.2
)

for seq in sequences:
    print("--------------------------")
    # Extract just the generated report part
    full_text = seq[0]['generated_text']
    print(f"{full_text}")
    print("--------------------------\n")