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
metric_titles = {
    "ttr":"Type–Token Ratio",
    "hapax":"Hapax Legomena Count",
    "adj_noun_ratio":"Adjective–Noun Ratio",
    "tree_depth":"Average Dependency Tree Depth",
    "fk_grade":"Flesch–Kincaid Grade Level"
}

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
    plt.title(metric_titles.get(metric,metric))
    filename=output/f"plot_{metric}.png"
    plt.savefig(filename)
    print(f"saved to {filename}")
    plt.close()

punc_cols=[
    "Semicolon Density",
    "Emdash Density",
    "Exclamation Density",
    "Question Density",
    "Comma Density",
    "Colon Density"
]

existing=[]
for col in punc_cols:
    if col in df.columns:
        existing.append(col)

if existing:
    punct_df=df.groupby("label")[existing].mean()
    plt.figure(figsize=(15,10))
    sb.heatmap(
        punct_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5
    )
    plt.title("Punctuation Density per 1,000 Words (Normalized)")
    plt.ylabel("Class")
    plt.xlabel("Punctuation Type")

    heatmap_path=output/"plot_punctuation_heatmap.png"
    plt.savefig(heatmap_path)
    print(f"saved to {heatmap_path}")
    plt.close()
