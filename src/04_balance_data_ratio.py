import random
from pathlib import Path

random.seed(42)

root=Path(__file__).parent.parent

path_human=root/"data"/"human"
path_generic=root/"data"/"generic_ai"
path_stylized=root/"data"/"stylized_ai"

human_files=list(path_human.glob("*.txt"))
numh=len(human_files)
print(f"human samples: {numh}")

numai=numh//2

def subsample(path,target):
    files=list(path.glob("*.txt"))
    print(f"{path.name}: {len(files)} → {target}")
    keep=random.sample(files,target)
    remove=set(files)-set(keep)
    for f in remove:
        f.unlink()

subsample(path_generic,numai)
subsample(path_stylized,numai)

print("ratios balanced")
