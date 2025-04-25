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

prompts = ["""
Ti si radijski prometni poročevalec. Na podlagi strukturiranih podatkov generiraj kratek, jedrnat prometni zapis.

Primer 1:
Podatki: Nesreča na avtocesti A1 med Celjem in Slovenskimi Konjicami. Zastoji v obe smeri.
Poročilo: Na avtocesti A1 med Celjem in Slovenskimi Konjicami je zaradi prometne nesreče prišlo do zastojev v obe smeri.

Primer 2:
Podatki: Zgoščen promet na ljubljanski obvoznici v smeri Šiške.
Poročilo: Na ljubljanski obvoznici je promet močno zgoščen v smeri Šiške.

Zdaj generiraj poročilo za:
Podatki: Prometna nesreča na primorski avtocesti med Uncem in Logatcem. Zaprt prehitevalni pas.
Poročilo:
"""]

sequences = pline(
    prompts,
    max_length=800,
    do_sample=False,          # allows randomness
    temperature=0.99,         # controls creativity
    top_k=1,                # limits candidates to top-k tokens
    repetition_penalty=1.2,  # helps fight exact repetition
    num_return_sequences=1,
    truncation=True
)


for seq in sequences:
    print("--------------------------")
    print(f"Result: {seq[0]['generated_text']}")
    print("--------------------------\n")