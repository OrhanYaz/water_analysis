# %% Import package
from collections import defaultdict

import nltk
import numpy as np
import pandas as pd
import seaborn as sb
import torch
import transformers
import xgboost
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    jaccard_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

# %% load Bert models
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
model_emb = transformers.BertModel.from_pretrained("bert-base-uncased")


# %% Load data
# English file
file_en = pd.read_csv("water_problem_nlp_en_for_Kaggle_100.csv", sep=";")
file_en.columns

# replace missing values in labels and convert into int
file_en = file_en.fillna(0)
file_en[file_en.columns[1:]] = file_en[file_en.columns[1:]].apply(
    lambda x: x.astype(int)
)

# %% data analysis

# Label correlation
corr = file_en[
    ["env_problems", "pollution", "treatment", "climate", "biomonitoring"]
].corr()
plt.figure(figsize=(12, 8))
sb.heatmap(corr, annot=True, cmap="YlGnBu")
plt.show()

contigency = pd.crosstab(file_en["env_problems"], file_en["pollution"])
plt.figure(figsize=(12, 8))
sb.heatmap(contigency, annot=True, cmap="YlGnBu")
plt.show()
res = chi2_contingency(contigency)
print(res)


# target balance
# environmental problems
print(file_en["env_problems"].value_counts())
ax = sb.countplot(x="env_problems", data=file_en)
plt.xlabel("env_problems")
plt.ylabel("Count")
plt.title("environmental problems")
for p in ax.patches:
    ax.annotate(
        format(p.get_height()),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 5),
        textcoords="offset points",
    )
plt.show()

# Pollution
print(file_en["pollution"].value_counts())
ax = sb.countplot(x="pollution", data=file_en)
plt.xlabel("pollution")
plt.ylabel("Count")
plt.title("pollution")
for p in ax.patches:
    ax.annotate(
        format(p.get_height()),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 5),
        textcoords="offset points",
    )
plt.show()

# treatment
print(file_en["treatment"].value_counts())
ax = sb.countplot(x="treatment", data=file_en)
plt.xlabel("treatment")
plt.ylabel("Count")
plt.title("treatment")
for p in ax.patches:
    ax.annotate(
        format(p.get_height()),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 5),
        textcoords="offset points",
    )
plt.show()

# climate
print(file_en["climate"].value_counts())
ax = sb.countplot(x="climate", data=file_en)
plt.xlabel("climate")
plt.ylabel("Count")
plt.title("climate")
for p in ax.patches:
    ax.annotate(
        format(p.get_height()),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 5),
        textcoords="offset points",
    )
plt.show()

# biomonitoring
print(file_en["biomonitoring"].value_counts())
ax = sb.countplot(x="biomonitoring", data=file_en)
plt.xlabel("biomonitoring")
plt.ylabel("Count")
plt.title("biomonitoring")
for p in ax.patches:
    ax.annotate(
        format(p.get_height()),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 5),
        textcoords="offset points",
    )
plt.show()


# %% Create embeddings
max_len = max([len(tokenizer.encode(text)) for text in file_en.text])
features = []
for text in file_en.text:
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    input_ids += [0] * (max_len - len(input_ids))
    input_ids_tensor = torch.tensor([input_ids])
    with torch.no_grad():
        segment_features = model_emb(input_ids_tensor)[0]
    features.append(segment_features.numpy().flatten())

text_embedded = np.vstack(pd.Series(features))

# Extract labels
Y = file_en.iloc[:, 1:]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    text_embedded, Y, train_size=0.85, random_state=42
)


# test single target
moc = MultiOutputClassifier(LogisticRegression())
moc.fit(X_train, y_train)

pred = moc.predict(X_test)

accuracy_score(y_test, pred)
multilabel_confusion_matrix(y_test, pred)
jaccard_score(y_test, pred, average="samples")

for i in y_test:
    print(i + ":", accuracy_score(y_test[i], pred[:, 0]))

# test multi label
from sklearn.multioutput import ClassifierChain

cc = ClassifierChain(LogisticRegression())
cc.fit(X_train, y_train)
pred = cc.predict(X_test)

accuracy_score(y_test, pred)
multilabel_confusion_matrix(y_test, pred)

# Ensemble classifier Chain
lr = LogisticRegression()
chains = [ClassifierChain(lr, order="random", random_state=i) for i in range(10)]
for chain in chains:
    chain.fit(X_train, y_train)

y_pred_chains = np.array([chain.predict(X_test) for chain in chains])
y_pred_ensemble = y_pred_chains.mean(axis=0)

# Jaccard
chain_jaccard_scores = [
    jaccard_score(y_test, y_pred_chain >= 0.5, average="samples")
    for y_pred_chain in y_pred_chains
]

ensemble_jaccard_score = jaccard_score(
    y_test, y_pred_ensemble >= 0.5, average="samples"
)

# Accuracy
chain_accuracy_scores = [
    accuracy_score(
        y_test,
        y_pred_chain >= 0.5,
    )
    for y_pred_chain in y_pred_chains
]

ensemble_qccuracy_score = accuracy_score(y_test, y_pred_ensemble >= 0.5)


# Classifier chain with adapted model
# using xgboost class_weight instead of over or undersampling
import xgboost as xgb
from sklearn.utils import class_weight

xgb_param = {
    "n_estimators": 500,
    "colsample_bytree": 0.7,
    "learning_rate": 0.17,
    "min_child_weight": 0.13,
    "max_depth": 20,
    "reg_alpha": 0.8,
    "reg_lambda": 0.9,
    "subsample": 0.75,
    "tree_method": "exact",
    "eval_metric": "logloss",
    "use_label_encoder": False,
}

# treatment with class_weight
classes_weights = class_weight.compute_sample_weight(
    class_weight="balanced", y=y_train["treatment"]
)
xgbC = xgb.XGBClassifier(**xgb_param)
xgbC.fit(X_train, y_train["treatment"], sample_weight=classes_weights)
y_treatment_pred = xgbC.predict(X_test)
accuracy_score(y_test["treatment"], y_treatment_pred)

# Classifier Chain with XGB
xgbC = xgb.XGBClassifier(**xgb_param)
cc = ClassifierChain(xgbC)
cc.fit(X_train, y_train)
pred = cc.predict(X_test)
for i in y_test:
    print(i + ":", accuracy_score(y_test[i], pred[:, 0]))


# treatment with class_weight
classes_weights = class_weight.compute_sample_weight(
    class_weight="balanced", y=y_train["climate"]
)
xgbC = xgb.XGBClassifier(**xgb_param)
xgbC.fit(X_train, y_train["climate"], sample_weight=classes_weights)
y_treatment_pred = xgbC.predict(X_test)
accuracy_score(y_test["climate"], y_treatment_pred)

# Ukranian file
file_ua = pd.read_csv(
    "water_problem_nlp_ua_for_Kaggle_100.csv", sep=";", encoding="windows-1251"
)
file_ua.columns

# replace missing values in labels and convert into int
file_ua = file_ua.fillna(0)
file_ua[file_ua.columns[1:]] = file_ua[file_ua.columns[1:]].apply(
    lambda x: x.astype(int)
)

# Translate text to english
from transformers import MarianMTModel, MarianTokenizer, Pipeline, pipeline

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-uk-en")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-uk-en")
translater = pipeline(
    task="translation",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
)
ls_translated = [i["translation_text"] for i in translater(file_ua.text.tolist())]
file_ua.text = ls_translated

# %% data analysis

# Label correlation
corr = file_ua[
    ["env_problems", "pollution", "treatment", "climate", "biomonitoring"]
].corr()
plt.figure(figsize=(12, 8))
sb.heatmap(corr, annot=True, cmap="YlGnBu")
plt.show()

contigency = pd.crosstab(file_ua["env_problems"], file_ua["pollution"])
plt.figure(figsize=(12, 8))
sb.heatmap(contigency, annot=True, cmap="YlGnBu")
plt.show()
res = chi2_contingency(contigency)
print(res)


# target balance
# environmental problems
print(file_ua["env_problems"].value_counts())
ax = sb.countplot(x="env_problems", data=file_ua)
plt.xlabel("env_problems")
plt.ylabel("Count")
plt.title("environmental problems")
for p in ax.patches:
    ax.annotate(
        format(p.get_height()),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 5),
        textcoords="offset points",
    )
plt.show()

# Pollution
print(file_ua["pollution"].value_counts())
ax = sb.countplot(x="pollution", data=file_ua)
plt.xlabel("pollution")
plt.ylabel("Count")
plt.title("pollution")
for p in ax.patches:
    ax.annotate(
        format(p.get_height()),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 5),
        textcoords="offset points",
    )
plt.show()

# treatment
print(file_ua["treatment"].value_counts())
ax = sb.countplot(x="treatment", data=file_ua)
plt.xlabel("treatment")
plt.ylabel("Count")
plt.title("treatment")
for p in ax.patches:
    ax.annotate(
        format(p.get_height()),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 5),
        textcoords="offset points",
    )
plt.show()

# climate
print(file_ua["climate"].value_counts())
ax = sb.countplot(x="climate", data=file_ua)
plt.xlabel("climate")
plt.ylabel("Count")
plt.title("climate")
for p in ax.patches:
    ax.annotate(
        format(p.get_height()),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 5),
        textcoords="offset points",
    )
plt.show()

# biomonitoring
print(file_ua["biomonitoring"].value_counts())
ax = sb.countplot(x="biomonitoring", data=file_ua)
plt.xlabel("biomonitoring")
plt.ylabel("Count")
plt.title("biomonitoring")
for p in ax.patches:
    ax.annotate(
        format(p.get_height()),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 5),
        textcoords="offset points",
    )
plt.show()


# Améliorations:
# tester autres data
# si données UK différentes de EN, concaténer les deux bases
# Cross validation avec un split stratitifié pour conserver le déséquilibre
# tester un classifier chain avec un model custom pour chaque target et pas le même modèle pour tous
# tester un stacked single target
# Tester de construire des target complètement décorrélés
# Tester une méthode avec réseaux bayésien (en utilisant d-separation) avec un bow comme text processing
