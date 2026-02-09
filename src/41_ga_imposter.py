import google.generativeai as genai
import time
import random
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from peft import PeftModel,PeftConfig

modelpath="models/distilbert_lora_tuned"
print(f"loading detector from {modelpath}")

config=PeftConfig.from_pretrained(modelpath)

basemodel=AutoModelForSequenceClassification.from_pretrained(
config.base_model_name_or_path,
num_labels=3
)

model=PeftModel.from_pretrained(basemodel,modelpath)

tokenizer=AutoTokenizer.from_pretrained(config.base_model_name_or_path)

if torch.cuda.is_available():
device="cuda"
else:
device="cpu"

model.to(device)
model.eval()

def gethumanscore(text):
inputs=tokenizer(
text,
return_tensors="pt",
truncation=True,
max_length=256,
padding=True
).to(device)

with torch.no_grad():
outputs=model(**inputs)
probs=F.softmax(outputs.logits,dim=-1).cpu().numpy()[0]

return probs[0]

geminiapikey="AIzaSyDewnm_ZVdI0vEgVlWMu1gssXHB9RAYpKc"
genai.configure(api_key=geminiapikey)

print("checking available gemini models")

availablemodels=[m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]

print(f"found {availablemodels}")

if "models/gemini-1.5-flash" in availablemodels:
modelname="models/gemini-1.5-flash"
elif "models/gemini-pro" in availablemodels:
modelname="models/gemini-pro"
else:
modelname=availablemodels[0]

print(f"using mutation model {modelname}")

mutatormodel=genai.GenerativeModel(modelname)

def mutatetext(text,currentscore,generation):
if currentscore<0.5:
prompt=(
"rewrite the following text to sound like a snippet from a classic 1920s novel hemingway style "
"use concrete sensory details sight sound smell "
"remove abstract concepts and replace them with physical descriptions "
"do not use words like therefore or in conclusion "
f"text {text}"
)
else:
prompt=(
"rewrite this text to be more human and casual "
"vary the sentence length add a sentence fragment for effect "
"use a contraction keep the meaning but break the robotic rhythm "
f"text {text}"
)

try:
response=mutatormodel.generate_content(
prompt,
generation_config=genai.types.GenerationConfig(temperature=0.9)
)
return response.text.strip()
except Exception:
print("api error")
return text

def rungeneticalgorithm(starttext,generations=5):
currenttext=starttext
print("starting evolution")
print(f"initial {currenttext[:80]}")

for gen in range(generations):
print(f"generation {gen+1}")

candidates=[]
print("mutating",end="")

for i in range(3):
mutant=mutatetext(currenttext,gethumanscore(currenttext),gen)
score=gethumanscore(mutant)
candidates.append((mutant,score))
print(" ",end="")
time.sleep(1.5)

print("done")

candidates.sort(key=lambda x:x[1],reverse=True)
besttext,bestscore=candidates[0]

print(f"best score {bestscore*100}")
print(f"text {besttext[:100]}")

currenttext=besttext

if bestscore>0.98:
print("stopping early")
break

return currenttext,bestscore

initialaiparagraph=(
"in conclusion the integration of artificial intelligence into daily workflows "
"represents a significant technological advancement it facilitates the automation "
"of repetitive tasks thereby allowing human operators to focus on complex problem solving"
)

finaltext,finalscore=rungeneticalgorithm(initialaiparagraph)

print("final evolved text")
print(finaltext)
print(f"final human score {finalscore*100}")
