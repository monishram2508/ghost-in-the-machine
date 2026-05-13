import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from peft import PeftModel,PeftConfig

modelpath="models/distilbert_lora_tuned"
print(f"loading model from {modelpath}")

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

print(f"model loaded on {device}")

id2label={0:"Human",1:"Generic AI",2:"Stylized AI"}

def predict(text):
    inputs=tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs=model(**inputs)
        logits=outputs.logits

    probs=F.softmax(logits,dim=-1).cpu().numpy()[0]

    predidx=probs.argmax()
    predlabel=id2label[predidx]
    confidence=probs[predidx]*100

    return predlabel,confidence,probs

print("tier c model")
print("type exit to stop")

while True:
    userinput=input("input paragraph ")
    if userinput.lower() in ["exit"]:
        break

    if len(userinput)<10:
        print("text too short try a full sentence")
        continue

    label,conf,allprobs=predict(userinput)

    print(f"prediction {label.lower()}")
    print(f"confidence {conf}")
    print(f"human {allprobs[0]*100}")
    print(f"generic ai {allprobs[1]*100}")
    print(f"stylized ai {allprobs[2]*100}")
