import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from peft import PeftModel, PeftConfig
from pathlib import Path
from matplotlib.patches import Patch

root=Path(__file__).parent.parent
path_plots=root/"data"/"plots"
path_model=root/"models"/"distilbert_lora_tuned"
print("loading model")

config=PeftConfig.from_pretrained(path_model)
base=AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path,num_labels=3)
model=PeftModel.from_pretrained(base,path_model)
tokenizer=AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model.to("cpu")
model.eval()
para="The Industrial Revolution was the transition to new manufacturing processes in Great Britain, continental Europe, and the United States, in the period from about 1760 to 1840. This transition included going from hand production methods to machines, new chemical manufacturing and iron production processes, the increasing use of steam power and water power, the development of machine tools, and the rise of the mechanized factory system."

# B. Generate Static Bar Charts
# --- SMART PLOTTER (Auto-Detects Winner) ---
def predict(texts):
    # SHAP may pass:
    # - list[str]
    # - list[list[str]]  (pre-tokenized)
    clean_texts=[]
    for t in texts:
        if isinstance(t,list):
            # tokens -> string
            clean_texts.append(tokenizer.convert_tokens_to_string(t))
        else:
            clean_texts.append(t)

    inputs=tokenizer(
        clean_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs=model(**inputs)
        probs=torch.softmax(outputs.logits,dim=1)

    return probs.cpu().numpy()

masker=shap.maskers.Text(tokenizer)
explainer=shap.Explainer(predict,masker,algorithm='permutation')

def plot_top_contributors(para):
    assert isinstance(para,str)
    shap_exp=explainer([para])
    values_all=shap_exp.values[0]
    tokens=tokenizer.tokenize(para)
    min_len=min(len(tokens),values_all.shape[0])
    tokens=tokens[:min_len]
    values_all=values_all[:min_len,:]
    total_impact=np.sum(values_all,axis=0)
    top_class=int(np.argmax(total_impact))
    classes=["Human","Generic AI","Stylized AI"]
    target_name=classes[top_class]

    # 3. Token contributions for winning class
    values=values_all[:,top_class]

    # 4. Top contributors
    idx=np.argsort(np.abs(values))[-30:]
    top_vals=values[idx]
    top_tokens=[tokens[i] for i in idx]

    # 5. Colors
    colors=["#ff0051" if v>0 else "#008bfb" for v in top_vals]

    # 6. Plot
    plt.figure(figsize=(10,6))
    plt.barh(range(len(top_vals)),top_vals,color=colors)
    plt.yticks(range(len(top_vals)),top_tokens,fontsize=12)
    plt.xlabel(f'SHAP Value (Impact on "{target_name}" Prediction)')
    plt.title(f'Why did the model choose "{target_name}"?',fontsize=14)
    plt.axvline(0,color="black",linewidth=0.8,linestyle="--")

    legend=[
        Patch(facecolor="#ff0051",label=f'Supports "{target_name}"'),
        Patch(facecolor="#008bfb",label=f'Opposes "{target_name}"')
    ]
    plt.legend(handles=legend,loc="lower right",fontsize=10)

    plt.tight_layout()
    filename=f"{path_plots}/shap_explanation_{target_name.replace(' ','_')}.png"
    plt.savefig(filename,dpi=300)
    plt.close()

    print(f'saved for "{target_name}" to {filename}')

plot_top_contributors(para)
