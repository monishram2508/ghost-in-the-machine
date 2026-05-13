import numpy as np 
from pathlib import Path

import torch
import torch.nn as nn 
from torch.utils.data import Dataset,DataLoader

from dotenv import load_dotenv

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
output=root/"data"/"plots"
output.mkdir(parents=True,exist_ok=True)

# GloVe configuration
path_glove=root/"data"/"dolma_300_2024_1.2M.100_combined.txt"
dim=300

def load_glove_embed(glove_file):
    embeddings={}
    with open(glove_file,'r',encoding='utf-8') as file:
        for line in tqdm(file,desc="loading GloVe"):
            val=line.strip().split()
            word=val[0]
            vector=np.array(val[1:],dtype='float32')
            embeddings[word]=vector
    return embeddings

#load GloVe
glove_embeddings=load_glove_embed(path_glove)
print(f"loaded {len(glove_embeddings)} GloVe vectors")

def extracttopicname(file):
    parts=file.name.split("_")
    return parts[-2]

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
else:
    dev="cpu"

device=torch.device(dev)
print(f"device={device}")

nnmodel=classifier(512,256,64,3,dim).to(device)

def preprocess_text(text):
    text=text.lower()
    cleaned=""
    for char in text:
        if char.isalnum() or char.isspace():
            cleaned+=char
        else:
            cleaned+=" "
    words=cleaned.split()
    return words

def get_glove_embeddings(text):
    words=preprocess_text(text)
    vectors=[]
    for word in words:
        if word in glove_embeddings:
            vectors.append(glove_embeddings[word])
    if len(vectors)==0:
        return np.zeros(dim)
    return np.mean(vectors,axis=0)

def embed_docs(path_map):
    emb,labels,filenames=[],[],[]
    for label in path_map.keys():
        path=path_map[label]
        files=list(path.glob("*.txt"))
        for file in tqdm(files,desc=f"  {label}"):
            text=file.read_text(encoding="utf-8")
            result=get_glove_embeddings(text)
            emb.append(result)
            labels.append(label)
            filenames.append(file.name)
    print(f"{len(emb)} docs embedded")
    return np.array(emb),np.array(labels),filenames

def file_quantity(path_map):
    c=0
    for label in path_map.keys():
        path=path_map[label]
        files=list(path.glob("*.txt"))
        for file in files:
            c+=1
    return c

#testing glove working
test=get_glove_embeddings("what is this task")
if test is not None and np.any(test):
    print(f"got {len(test)} dimensions")
else:
    print("failed")

path_map={
    "human":root/"data"/"human",
    "generic":root/"data"/"generic_ai",
    "stylized_ai":root/"data"/"stylized_ai"
}
#avoid recomputing
cache_path=root/"data"/"glove_embeddings_cache.npz"

if cache_path.exists():
    print(f"\nloading cached embeddings from {cache_path}")
    cached=np.load(cache_path,allow_pickle=True)
    embeds=cached['embeddings']
    labels=cached['labels']
    filenames=cached['filenames']
else:
    print("\nembedding documents with GloVe...")
    embeds,labels,filenames=embed_docs(path_map)
    np.savez(cache_path,embeddings=embeds,labels=labels,filenames=filenames)
    print(f"cached embeddings to {cache_path}")

encode_label=LabelEncoder()
y=encode_label.fit_transform(labels)

topics=np.array([extracttopicname(Path(f)) for f in filenames])
print("unique topics:",len(np.unique(topics)))
unique_topics=np.unique(topics)
rng=np.random.default_rng(42)
rng.shuffle(unique_topics)

test_topics=set(unique_topics[:int(0.2*len(unique_topics))])

train_mask=~np.isin(topics,list(test_topics))
test_mask=np.isin(topics,list(test_topics))

x_train=embeds[train_mask]
x_test=embeds[test_mask]

y_train=y[train_mask]
y_test=y[test_mask]

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

from sklearn.linear_model import LogisticRegression

print("\nlinear baseline (logistic regression)")

lin_clf=LogisticRegression(
    max_iter=3000,
    multi_class="auto",
    n_jobs=-1
)

lin_clf.fit(x_train,y_train)

lin_preds=lin_clf.predict(x_test)

lin_acc=accuracy_score(y_test,lin_preds)

print(f"linear test accuracy:{lin_acc*100:.2f}%\n")

print(classification_report(
    encode_label.inverse_transform(y_test),
    encode_label.inverse_transform(lin_preds)
))

