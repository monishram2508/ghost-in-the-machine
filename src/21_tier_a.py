from pathlib import Path
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb 
import seaborn as sb 
import matplotlib.pyplot as plt

root=Path(__file__).parent.parent
file=root/"data"/"fingerprint_data.csv"
df=pd.read_csv(file)

metrics=["ttr","hapax","adj_noun_ratio","tree_depth","fk_grade"]
print("total s  amples=",len(df))
df[metrics]=df[metrics].apply(pd.to_numeric,errors="coerce")

x=df[metrics].values
y_label=df["label"].values
label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y_label)

print("label mapping:")
for i,label in enumerate(label_encoder.classes_):
    print(f"{label} is {i}")

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
print(f"data split train = {len(x_train)} test={len(x_test)} total={len(df)}")


print("starting model creation")
model=xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=3,
    random_state=42,
    eval_metric="mlogloss",
    n_jobs=-1,
    verbosity=1
)
model.fit(x_train,y_train)
print("training complete")

y_pred=model.predict(x_test)
acc=accuracy_score(y_test,y_pred)
print(acc)

y_test_text=label_encoder.inverse_transform(y_test)
y_pred_text=label_encoder.inverse_transform(y_pred)

print(classification_report(y_test_text,y_pred_text))

cm=confusion_matrix(y_test_text,y_pred_text,labels=label_encoder.classes_)
plt.figure(figsize=(10,6))
sb.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("tier a model: xgboost confusion_matrix")
output=root/"data"/"plots"
output.mkdir(parents=True,exist_ok=True)
plt.savefig(output/"tier_a_confusion_matrix.png")
plt.close()
print("confusion matrix saved")
