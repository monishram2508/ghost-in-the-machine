import json
import spacy
from pathlib import Path
from collections import Counter
import math

nlp=spacy.load("en_core_web_sm",disable=["parser","ner"])
root=Path(__file__).parent.parent
path_chunked=root/"data"/"human"
path_frequencies=root/"data"/"frequencies"
path_topics=root/"data"/"topics"

def extract_topics(path_chunked,path_frequencies):
    path_frequencies.mkdir(parents=True,exist_ok=True)
    files=sorted(path_chunked.glob("*.txt"))
    book_files={}
    for file in files:
        base="_".join(file.stem.split("_")[:-2])
        if base not in book_files:
            book_files[base]=[]
        book_files[base].append(file)
    
    book_nouns = {}
    for book,file_list in book_files.items():
        print(f"Processing {book} ({len(file_list)} chunks).")
        texts=[f.read_text(encoding="utf-8") for f in file_list]
        noun_counter=Counter()
        for i,doc in enumerate(nlp.pipe(texts, batch_size=50),1):
            for token in doc:
                if token.pos_ in ("NOUN","PROPN"):
                    if len(token.text) > 3 and token.is_alpha:
                        noun_counter[token.text.lower()]+=1
            print(f"{i}/{len(texts)} chunks",end="\r")
        book_nouns[book]=dict(noun_counter)
        freq_path=path_frequencies/f"{book}_frequencies.json"
        with open(freq_path,"w",encoding="utf-8") as f:
            json.dump(dict(noun_counter),f,indent=2)
    return book_nouns

def extract_frequencies(path_chunked, path_frequencies):
    path_frequencies.mkdir(parents=True, exist_ok=True)
    files = sorted(path_chunked.glob("*.txt"))
    book_files = {}
    for file in files:
        base = "_".join(file.stem.split("_")[:-2])
        if base not in book_files:
            book_files[base] = []
        book_files[base].append(file)
    
    book_nouns = {}
    for book, file_list in book_files.items():
        print(f"Processing {book} ({len(file_list)} chunks).")
        texts = [f.read_text(encoding="utf-8") for f in file_list]
        noun_counter = Counter()
        for i, doc in enumerate(nlp.pipe(texts, batch_size=50), 1):
            for token in doc:
                if token.pos_ in ("NOUN", "PROPN"):
                    if len(token.text) > 3 and token.is_alpha:
                        noun_counter[token.text.lower()] += 1
            print(f"{i}/{len(texts)} chunks", end="\r")
        print()
        book_nouns[book] = Counter(noun_counter)
        freq_path = path_frequencies / f"{book}_frequencies.json"
        with open(freq_path, "w", encoding="utf-8") as f:
            json.dump(dict(noun_counter), f, indent=2)
    return book_nouns

def calculate_tfidf(book_nouns):
    doc_frequency = Counter()
    for book, noun_counts in book_nouns.items():
        for noun in noun_counts.keys():
            doc_frequency[noun] += 1
    
    total_books = len(book_nouns)
    book_tfidf = {}
    
    for book, noun_counts in book_nouns.items():
        tfidf_scores = {}
        total_nouns = sum(noun_counts.values())
        
        for noun, count in noun_counts.items():
            tf = count / total_nouns
            idf = math.log(total_books / doc_frequency[noun])
            tfidf_scores[noun] = tf * idf
        
        book_tfidf[book] = tfidf_scores
    
    return book_tfidf


def save_topics(book_tfidf, path_topics, top_n=10):
    path_topics.mkdir(parents=True, exist_ok=True)
    
    for book, tfidf_scores in book_tfidf.items():
        sorted_nouns = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        top_words = [word for word, score in sorted_nouns[:top_n]]
        
        output_path = path_topics / f"{book}_topics.txt"
        output_path.write_text('\n'.join(top_words), encoding='utf-8')


if __name__ == "__main__":
    book_nouns = extract_frequencies(path_chunked, path_frequencies)
    book_tfidf = calculate_tfidf(book_nouns)
    save_topics(book_tfidf, path_topics, top_n=10)