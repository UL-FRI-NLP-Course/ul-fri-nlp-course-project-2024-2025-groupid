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
    {"role": "user", "content": """Natančno pretvori prometne podatke v uradna prometna poročila po naslednjih pravilih:

1. Ne dodajaj informacij, ki niso navedene v podatkih
2. Uporabi uradna imena cest 
   - LJUBLJANA-KOPER: primorska avtocesta proti Kopru ali Ljubljani
   - LJUBLJANA-KARAVANKE: gorenjska avtocesta proti Karavankam/Avstriji ali Ljubljani
   - LJUBLJANA-MARIBOR: štajerska avtocesta proti Mariboru ali Ljubljani
   - MARIBOR-GRUŠKOVJE: podravska avtocesta proti Lendavi/Madžarski ali Mariboru
   - Gorenjska avtocesta proti Kranju = gorenjska avtocesta proti Avstriji/Karavankam
   - Dolenjska avtocesta proti Novemu mestu = dolenjska avtocesta proti Hrvaški/Obrežju
   - Hitra cesta skozi Maribor = regionalna cesta Betnava-Pesnica / nekdanja hitra cesta skozi Maribor
   - Hitra cesta razcep Nanos-Vrtojba = vipavska hitra cesta v smeri proti Italiji/Vrtojbi
3. Za vsak dogodek uporabi eno od formul:
   - Cesta in smer + razlog + posledica in odsek
   - Razlog + cesta in smer + posledica in odsek

4. Počasni pas je pas za počasna vozila.

Polovična zapora ceste pomeni: promet je tam urejen izmenično enosmerno. 

Vsi pokriti vkopi itd. so predori, razen galerija Moste ostane galerija Moste.

Ko se kaj dogaja na razcepih, je treba navesti od kod in kam: Na razcepu Kozarje je zaradi nesreče oviran promet iz smeri Viča proti Brezovici, …

Pri obvozu: Obvoz je po vzporedni regionalni cesti/po cesti Lukovica-Blagovica ali vozniki se lahko preusmerijo na vzporedno regionalno cesto 

PRIMER 1:
PODATKI: Nesreča na avtocesti A1 med Celjem in Slovenskimi Konjicami. Zastoji v obe smeri.
POROČILO: Na avtocesti A1 med Celjem in Slovenskimi Konjicami je zaradi prometne nesreče prišlo do zastojev v obe smeri.

PRIMER 2:
PODATKI: Zgoščen promet na ljubljanski obvoznici v smeri Šiške.
POROČILO: Na ljubljanski obvoznici je promet močno zgoščen v smeri Šiške.

ZDAJ PRETVORI SLEDEČE PODATKE V POROČILO:
PODATKI: POMEMBNO: Previdno vozite po štajerski avtocesti od Šempetra proti Žalcu torej proti Mariboru. Na vozišču je voznik, ki vozi v napačno smer. 
ZASTOJI: Zastoj je na Bledu. Zastoj je na cesti Bled - Lesce. 
OVIRE: Zaradi vozila v okvari je zaprt en pas na cesti Črnuče - Tomačevo, proti krožišču Tomačevo. 
Zaradi vozila v okvari je zaprt en pas na cesti Črnuče - Ljubljana, proti krožišču Tomačevo. 
MEJNI PREHODI: Čakalne dobe pri vstopu: Slovenska vas, Obrežje, Dobovec in Gruškovje. Čakalne dobe pri vstopu: Metlika, Slovenska vas, Obrežje in Dobovec. 
POROČILO:
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