import spacy
import textstat
import pandas as pd
import os
from collections import Counter
from pathlib import Path

nlp=spacy.load("en_core_web_sm")
root=Path(__file__).parent.parent
folders={
    "human":root/"data"/"human",
    "generic_ai":root/"data"/"generic_ai",
    "stylized_ai":root/"data"/"stylized_ai"
}

def ttr_hapax(doc):
    if len(doc)==0: return None,None
    words=[]
    for token in doc:
        if token.is_alpha:
            words.append(token.text.lower())
    if len(words)==0: return None,None
    unique_words=set(words)
    num_unique=len(unique_words)
    ttr=num_unique/len(words)
    words=Counter(words)
    sum=0
    for freq in words.values():
        if freq==1:
            sum+=1
    return (ttr,sum)

def pos_dist(doc):
    noun=[]
    adj=[]
    for token in doc:
        if token.pos_=="NOUN":
            noun.append(token.text.lower())
        if token.pos_=="ADJ":
            adj.append(token.text.lower())
    if len(noun)>0: return len(adj)/len(noun)
    else: return 0

def depedency_depth(doc):
    depths=[]
    for sentence in doc.sents:
        root=[]
        for token in sentence:
            if token.head==token:
                root.append(token)        
        if not root: continue
        root_main=root[0]
        def get_depth(node):
            if not list(node.children):
                return 1
            return 1+max(get_depth(child) for child in node.children)
        depths.append(get_depth(root_main))
    if depths: return sum(depths)/len(depths)
    else: return 0

def grade(text):
    return textstat.flesch_kincaid_grade(text)

def process():
    data=[]
    for label,folder in folders.items():
        print(label,folder)
        for file in tdqm(folder.glob("*.txt"),desc=f"Iterating through files in {label}"):
            print(file.name)
            text=file.read_text(encoding="utf-8")
            doc=nlp(text)
            ttr,hapax=ttr_hapax(doc)
            if ttr==None: continue
            pos=pos_dist(doc)
            depth=depedency_depth(doc)
            fk_grade=grade(text)
            punc=punctuation_density(text)
            row={
                "filename":file.stem,
                "label":label,
                "ttr":ttr,
                "hapax":hapax,
                "adj_noun_ratio":pos,
                "tree_depth":depth,
                "semicolon_density": punc["semicolon"],
                "emdash_density": punc["emdash"] + punc["dash"],
                "exclamation_density": punc["exclamation"],
                "question_density": punc["question"],
                "comma_density": punc["comma"],
                "colon_density": punc["colon"],
                "fk_grade":fk_grade
            }
            data.append(row)
            if data:
                df=pd.DataFrame(data)
                path=root/"data"/"fingerprint_data.csv"
                df.to_csv(path,index=False)
                print("done")
    return path

def punctuation_density(text):
    punctuations={
        "semicolon":";",
        "dash":"--",
        "emdash":"-",
        "exclamation":"!",
        "question":"?",
        "comma":",",
        "colon":":"
    }
    words=text.split()
    word_count=len(words)
    if word_count==0:
        for k in punctuations:
            punctuations[k]=0
        return punctuations
    densities={}
    for name,symbol in punctuations.items():
        count=text.count(symbol)
        densities[name]=count/word_count*1000
    return densities

def display_results(file):
    df=pd.read_csv(file)
    print(df)

if __name__=="__main__":
    file=process()
    display_results(file)
