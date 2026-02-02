import re
from pathlib import Path 
import string
from collections import Counter

root=Path(__file__).parent.parent
path_raw=root/"data"/"raw"
path_cleaned=root/"data"/"clean"
path_chunked=root/"data"/"human"
path_topics=root/"data"/"topic"

def tokenize(text,keywords):
    stopwords={
        'the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'was','for', 'on', 'with', 'as', 'at', 'be', 'this', 'by', 'from', 'or','an', 'but', 'not', 'are', 'have', 'had', 'has', 'were', 'been','their', 'they', 'he', 'she', 'we', 'you', 'me', 'my', 'his', 'her','him', 'them', 'would', 'could', 'should', 'will', 'can', 'may','must', 'said', 'one', 'two', 'all', 'when', 'there', 'which','who', 'what', 'where', 'how', 'why', 'if', 'so', 'up', 'out','about', 'into', 'than', 'then', 'now', 'only', 'its', 'no', 'yes','very', 'just', 'much', 'more', 'some', 'such', 'like', 'too','also', 'even', 'well', 'back', 'through', 'after', 'before','between', 'under', 'again', 'over', 'any', 'both', 'each', 'few','other', 'another', 'same', 'own'
    }
    text=text.lower()
    text=text.translate(str.maketrans("","",string.punctuation))
    words=text.split()
    for word in words:
        if word not in stopwords and len(word)>2:
            keywords[word]+=1
    return keywords

def chunk(path_cleaned):
    for file in path_cleaned.glob("*.txt"):
        text=file.read_text(encoding="utf-8")
        chunks=chunkpara(text,500)
        path_chunked.mkdir(parents=True,exist_ok=True)
        for i,chunk in enumerate(chunks):
            base=file.stem.split("_")
            base="_".join(base[:-1])
            write_path=path_chunked/f"{base}_chunk_{i}.txt"
            write_path.write_text(chunk,encoding="utf-8")

def read(path_raw):
    for file in path_raw.glob("*.txt"):
        cleanlines=clean(file)
        text="\n".join(cleanlines)
        path_cleaned.mkdir(parents=True,exist_ok=True)
        write_path=path_cleaned/f"{file.stem}_cleaned.txt"
        write_path.write_text(text,encoding="utf-8")

def splitsentence(sentence):
    return re.split(r'(?<=[.!?])\s+',sentence)

def chunkpara(text,maxwords=500):
    sentences=splitsentence(text)
    chunks=[]
    current=[]
    count=0
    for sentence in sentences:
        words=sentence.split()
        current.append(sentence)
        count+=len(words)
        if count>=maxwords:
            chunks.append(" ".join(current).strip())
            current=[]
            count=0
    if current:
        chunks.append(" ".join(current).strip())
    return chunks

def findstart(lines):
    for i in range(len(lines)-2):
        check=lines[i:i+3]
        if all(
            len(line)>50
            and any(c.islower() for c in line)
            and "--" not in line
            for line in check
        ):
            return i
    return 0

def normalize(lines):
    cleaned=[]
    prev_blank=False
    for line in lines:
        line=line.strip()
        if not line:
            if not prev_blank:
                cleaned.append("")
            prev_blank=True
        else:
            cleaned.append(line)
            prev_blank=False
    return cleaned

def clean(path_input):
    with open(path_input,'r',encoding='utf-8') as text:
        f=text.read()
        lines=f.splitlines()
        start=0
        end=len(lines)
        for i,line in enumerate(lines):
            if "*** START" in line.upper():
                start=i+1
            if "*** END" in line.upper():
                end=i
                break
        lines=lines[start:end]
        lines=lines[findstart(lines):]
        lines=normalize(lines)
        return lines

def isimage(line):
    lower=line.lower()
    if ".jpg" in lower or ".png" in lower or ".jpeg" in lower or "illustration" in lower:
        return True

def tocline(line):
    if "chapter" in line:
        return True
    
def allcapsline(line):
    letter=False
    for i in line:
        if i.isalpha():
            letter=True
            if i.islower():
                return False
    return letter

def dashheavy(line):
    if line.count("--")>=2:
        return True
    return False

if __name__=="__main__":
    read(path_raw)
    chunk(path_cleaned)