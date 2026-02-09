from importlib import metadata
import time
import os
import pandas as pd
import numpy as np 
from pathlib import Path

import torch
import torch.nn as nn 
from torch.utils.data import Dataset,DataLoader

import chromadb
from chromadb.config import Settings
from google import genai as gemini
from google.genai import types
from dotenv import load_dotenv
from transformers import AutoTokenizer,AutoModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt 
import seaborn as sb 
from tqdm import tqdm

load_dotenv()
root=Path(__file__).parent.parent
file=root/"data"/"fingerprint_data.csv"
path_chroma=root/"data"/"chromadb"
output=root/"data"/"plots"
path_chroma.mkdir(parents=True,exist_ok=True)
output.mkdir(parents=True,exist_ok=True)

client=gemini.Client(api_key=os.getenv("GEMINI_API_KEY"))
# for model in client.models.list():
#     print(model.name)
print("gemini loaded")
gem_model="models/gemini-embedding-001"
dimensions=768
batch_size=32

class classifier(nn.Module):
    def __init__(self,h1,h2,h3,num_classes,dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(dim,h1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(h1,h2),
            nn.ReLU(),
            nn.Linear(h2,h3),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(h3,num_classes),
        )
    def forward(self,x):
        return self.net(x)

class texttotorch(Dataset):
    def __init__(self,x,y):
        self.x=torch.FloatTensor(x)
        self.y=torch.LongTensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self,index):
        return self.x[index],self.y[index]

if torch.cuda.is_available():
    dev="cuda"
else: dev="cpu"

device=torch.device(dev)
print(f"device={device}")

nnmodel=classifier(512,256,64,3,768).to(device)

print("init chromadb")
chroma_db=chromadb.PersistentClient(path=str(path_chroma))
coll=chroma_db.get_or_create_collection(name="gemini_embeddings",metadata={"desc":"gemini embeddings for ai detection"})
print(f"existing docs:{coll.count()}")

def get_gemini_embeddings(text):
    for attempt in range(5):
        try:
            res=client.models.embed_content(model=gem_model,contents=text)
            result=np.array(res.embeddings[0].values)
            return result
        except Exception as e:
            print(e)

def embed_docs(path_map,collection):
    emb=[]
    ids=[]
    md=[]
    size=100
    dc=0
    for label in path_map.keys():
        path=path_map[label]
        files=list(path.glob("*.txt"))
        for file in tqdm(files,desc=f"  {label}"):
            text=file.read_text(encoding="utf-8")
            result=get_gemini_embeddings(text)
            ids.append(str(dc))
            emb.append(result.tolist())
            md.append({"filename":file.name,"label":label})
            dc+=1 
            time.sleep(0.1)
            if len(ids)>=size:
                coll.add(ids=ids,embeddings=emb,metadatas=md)
                ids=[]
                emb=[]
                md=[]
    if ids:
        coll.add(ids=ids,embeddings=emb,metadatas=md)
    print("store all the data in chromadb")
    print(f"{coll.count()} docs embedded")

def file_quantity(path_map):
    c=0
    for label in path_map.keys():
        path=path_map[label]
        files=list(path.glob("*.txt"))
        for file in files:
            c+=1
    return c
        
#testing gemini working
test=get_gemini_embeddings("what is this task")
if test is not None:
    print(f"got {len(test)}")
else:
    print("failed")

path_map={
    "human":root/"data"/"human",
    "generic":root/"data"/"generic_ai",
    "stylized_ai":root/"data"/"stylized_ai"
}

total_files=file_quantity(path_map)
if coll.count()>=total_files:
    print(f"\nfound {coll.count()} embeddings")
else:
    embed_docs(path_map,coll)

print("loading embedded data")
data=coll.get(include=["embeddings","metadatas"])
embeds=np.array(data["embeddings"])
y=[]
for md in data["metadatas"]:
    y.append(md["label"])
labels=np.array(y)

encode_label=LabelEncoder()
y=encode_label.fit_transform(labels)

x_train,x_test,y_train,y_test=train_test_split(embeds,y,test_size=0.2,random_state=42,stratify=labels)
print("\ndata split")
print(f"train:test :: {len(x_train)}:{len(x_test)}")

load_train=DataLoader(
    texttotorch(x_train,y_train),
    batch_size=32,
    shuffle=True
)
load_test=DataLoader(
    texttotorch(x_test,y_test),
    batch_size=32,
    shuffle=True
)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(nnmodel.parameters(),lr=0.001)
print("\ntraining")
train_loss,train_acc=[],[]
for epoch in range(20):
    nnmodel.train()
    tloss=0
    correct=0
    total=0
    for bx,by in load_train:
        bx,by=bx.to(device),by.to(device)
        optimizer.zero_grad()
        outputs=nnmodel(bx)
        loss=criterion(outputs,by)
        loss.backward()
        optimizer.step()
        tloss+=loss.item()
        _,predicted=torch.max(outputs,1)
        correct+=(predicted==by).sum().item()
        total+=by.size(0)
    avgloss=tloss/len(load_train)
    acc=correct/total
    train_loss.append(avgloss)
    train_acc.append(acc)
    print(f"epoch no. {epoch+1} loss {avgloss:.4f} acc {acc*100:.2f}%")

print("completed training")

###########################################
nnmodel.eval()
all_preds,all_true=[],[]
with torch.no_grad():
    for bx,by in load_test:
        bx=bx.to(device)
        outputs=nnmodel(bx)
        _, predicted=torch.max(outputs,1)
        all_preds.extend(predicted.cpu().numpy())
        all_true.extend(by.numpy())

test_acc=accuracy_score(all_true,all_preds)

print("\nresults:")
print(f"test acc:{test_acc*100}\n")

print(classification_report(
    encode_label.inverse_transform(all_true),
    encode_label.inverse_transform(all_preds)
))

cm=confusion_matrix(all_true,all_preds)
plt.figure(figsize=(10,6))
sb.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues",
    xticklabels=encode_label.classes_,
    yticklabels=encode_label.classes_
)
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title(f"confusion matrix\ntest accuracy: {test_acc:.2%}")
plt.savefig(output/"tier_b_confusion_matrix.png")
plt.close()

print("diagnosis")
#checking embedding
print("\nchecking embedding similarity:")
train_test_sim=cosine_similarity(x_train[:50],x_test[:50])
max_sims=train_test_sim.max(axis=1)
print(f"max similarity per sample: {max_sims.mean():.4f}")
print(f"samples with >99% similarity: {(max_sims>0.99).sum()}")

#checking class separation
print("\nchecking class separation: avg within class similarity")
for i,label in enumerate(encode_label.classes_):
    class_mask=y==i
    class_embeddings=embeds[class_mask]
    within_sim=cosine_similarity(class_embeddings[:20],class_embeddings[:20])
    avg_within=within_sim[np.triu_indices(20, k=1)].mean()
    print(f"{label}: {avg_within:.4f}")

#checking file count
print("\nfile count:")
for label,path in path_map.items():
    count=len(list(path.glob("*.txt")))
    print(f"{label}:{count} files")

#random prediction
print("\nrandom test prediction:")
index=np.random.randint(len(x_test))
x_sample=torch.FloatTensor(x_test[index]).unsqueeze(0).to(device)
with torch.no_grad():
    output=nnmodel(x_sample)
    probs=torch.softmax(output,dim=1)
print(f"true label: {encode_label.classes_[y_test[index]]}")
print(f"probabilities: {probs[0].cpu().numpy()}")
