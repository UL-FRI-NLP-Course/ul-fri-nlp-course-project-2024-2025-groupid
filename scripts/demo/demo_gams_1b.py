from transformers import pipeline
import torch
print(torch.cuda.is_available())   # Should return True
print(torch.cuda.get_device_name(0))  # Optional: shows your GPU name

model_id = ("cjvt/GaMS-1B")

pline = pipeline(
    "text-generation",
    model=model_id,
    device_map={"": 0},
    torch_dtype=torch.float16
)

prompts = ["""Natančno pretvori prometne podatke v uradna prometna poročila po naslednjih pravilih:

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
"""]

sequences = pline(
    prompts,
    max_new_tokens=300,      # Control the length more precisely
    do_sample=True,
    temperature=0.8,         # Slightly lower for more coherence
    top_k=2,                # Allow more token options
    top_p=0.99,              # Add nucleus sampling
    repetition_penalty=1.2,
)


for seq in sequences:
    print("--------------------------")
    print(f"Result: {seq[0]['generated_text']}")
    print("--------------------------\n")