import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sb
import pandas as pd
from pathlib import Path

root=Path(__file__).parent.parent
file=root/"data"/"fingerprint_data.csv"
output=root/"data"/"plots"
output.mkdir(parents=True,exist_ok=True)

df=pd.read_csv(file)
metrics=["ttr","hapax","adj_noun_ratio","tree_depth","fk_grade"]

for col in metrics:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

sb.set_theme(style="whitegrid")

for metric in metrics:
    fig,ax=plt.subplots(figsize=(10,6),facecolor="white")
    if metric=="fk_grade":
        sb.violinplot(data=df,x="label",y=metric,palette="viridis",inner="quartile")
    else:
        sb.boxplot(data=df,x="label",y=metric,palette="viridis")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.title(f"Comparison of {metric.title()}")
    filename=output/f"plot_{metric}.png"
    plt.savefig(filename)
    print(f"saved to {filename}")
    plt.close()
