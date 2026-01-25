from pathlib import Path 

root=Path(__file__).parent.parent
path_raw=root/"data"/"raw"
path_cleaned=root/"data"/"clean"
path_chunked=root/"data"/"human"

def read(path_raw):
    for file in path_raw.glob("*.txt"):
        cleanlines=clean(file,path_cleaned)
        cleanlines=format(cleanlines)

def clean(path_input, path_output):
    with open(path_input,'r',encoding='utf-8') as text:
        f=text.read()
        lines=f.splitlines()
        for i,line in enumerate(lines):
            if "*** START" in line.upper():
                start=i+1
            if "*** END" in line.upper():
                end=i
                break
        cleanlines=lines[start:end]
        return cleanlines
    
def format(lines):
    cleanlines=[]
    for line in lines:
        # print(line)
        line=line.strip()
        # we need to check for whitespaces
        
        if line=="":
            continue
        else:
            cleanlines.append(line)
    print(cleanlines[:50])
    return cleanlines

if __name__=="__main__":
    read(path_raw)