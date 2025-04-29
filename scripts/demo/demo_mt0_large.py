from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
# from huggingface_hub import login

# login(token="hf_your_token_here")

# Load model
model_id = "bigscience/mt0-large"

print("CUDA available:", torch.cuda.is_available())

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    # load_in_8bit=True
)

# Create text-generation pipeline
pline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float16
)

# Define the prompt
prompt = """Ustvari prometno poročilo iz podanih podatkov.

Strogo upoštevaj ta pravila:

1. FORMULACIJA POROČILA:
   - Uporabi eno od dveh formul: "Cesta in smer + razlog + posledica in odsek" ALI "Razlog + cesta in smer + posledica in odsek"
   - Primer: "Na štajerski avtocesti proti Mariboru je zaradi nesreče oviran promet med priključkom Celje-vzhod in počivališčem Zima."

2. PRAVILNA IMENA AVTOCEST IN SMERI:
   - LJUBLJANA-KOPER: primorska avtocesta proti Kopru ali Ljubljani
   - LJUBLJANA-KARAVANKE: gorenjska avtocesta proti Karavankam/Avstriji ali Ljubljani
   - LJUBLJANA-MARIBOR: štajerska avtocesta proti Mariboru ali Ljubljani
   - MARIBOR-GRUŠKOVJE: podravska avtocesta proti Lendavi/Madžarski ali Mariboru
   - Gorenjska avtocesta proti Kranju = gorenjska avtocesta proti Avstriji/Karavankam
   - Dolenjska avtocesta proti Novemu mestu = dolenjska avtocesta proti Hrvaški/Obrežju
   - Hitra cesta skozi Maribor = regionalna cesta Betnava-Pesnica / nekdanja hitra cesta skozi Maribor
   - Hitra cesta razcep Nanos-Vrtojba = vipavska hitra cesta v smeri proti Italiji/Vrtojbi

3. LJUBLJANSKA OBVOZNICA:
   - Vzhodna: razcep Malence (proti Novemu mestu), razcep Zadobrova (proti Mariboru)
   - Zahodna: razcep Koseze (proti Kranju), razcep Kozarje (proti Kopru)
   - Severna: razcep Koseze (proti Kranju), razcep Zadobrova (proti Mariboru)
   - Južna: razcep Kozarje (proti Kopru), razcep Malence (proti Novemu mestu)

4. TERMINOLOŠKE ZAHTEVE:
   - Počasni pas → pas za počasna vozila
   - Polovična zapora ceste → promet je urejen izmenično enosmerno
   - Zaprta polovica avtoceste → promet je urejen le po polovici avtoceste v obe smeri
   - Pokriti vkopi → predori (razen galerija Moste)

Poročilo začni z "Podatki o prometu." in strogo sledi hierarhiji dogodkov. Posamezne dogodke navedi v jedrnatih stavkih, brez dodatnih razlag ali komentarjev. Vključi vse pomembne podatke, a brez ponavljanja. 

PRIMER PRAVILNEGA POROČILA:
Za podatke:
- Na štajerski avtocesti proti Mariboru je zaradi nesreče oviran promet med priključkom Celje-vzhod in počivališčem Zima.
- Zaradi burje je na vipavski hitri cesti proti Vrtojbi prepovedan promet za hladilnike in vsa vozila s ponjavami.
- Zaradi vozila v okvari na južni ljubljanski obvoznici med priključkoma Koseze in Tomačevo nastaja zatoj.
- Na mejnih prehodih Obrežje in Slovenska vas je čakalna doba eno uro.

Pravilno poročilo bi bilo:
"Podatki o prometu.
Zaradi vozila v okvari je zaprt en pas na južni ljubljanski obvoznici med razcepi Kozarje in Malence proti Novemu mestu.
Pri vstopu na mejnem prehodu Obrežje je čakalna doba."

PODATKI: Previdno vozite po štajerski avtocesti od Šempetra proti Žalcu torej proti Mariboru. Na vozišču je voznik, ki vozi v napačno smer. 
         Zastoj je na Bledu. Zastoj je na cesti Bled - Lesce. 
         Zaradi vozila v okvari je zaprt en pas na cesti Črnuče - Tomačevo, proti krožišču Tomačevo. 
         Čakalne dobe pri vstopu: Slovenska vas, Obrežje, Dobovec in Gruškovje. Čakalne dobe pri vstopu: Metlika, Slovenska vas, Obrežje in Dobovec. 

POROČILO:
"""

# Generate output
sequences = pline(
    prompt,
    max_new_tokens=500,
    do_sample=True,
    repetition_penalty=1.2,
    num_beams=4
)

for seq in sequences:
    print("--------------------------")
    print(seq)
    print("--------------------------\n")