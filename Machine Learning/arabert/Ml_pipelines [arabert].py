import os
import sys

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(r"../../sysPath")
os.chdir(dname)

models_dir = os.path.join(dname, 'Trained Models [RNN]')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

import pickle
import numpy as np
import pandas as pd
import preProcessData  #type: ignore
from transformers import TFAutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report # Accuracy metrics 

from summarize import saveFig # type: ignore
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier



tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabert")
model = TFAutoModel.from_pretrained("aubmindlab/bert-base-arabert")
batch_size = 64



def extract_arabert_features(texts, model, tokenizer):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")
    outputs = model(inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Using the embeddings of the [CLS] token



def process_in_batches(texts, batch_size):
    features = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_features = extract_arabert_features(batch_texts, model, tokenizer)
        features.append(batch_features)
        print(f"batch {i} of {len(texts)} Done")
    return np.concatenate(features, axis=0)



ArsarcasmTrain = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/training_data.csv")[["tweet", "dialect", "sarcasm"]]
ArsarcasmTest = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/testing_data.csv")[["tweet", "dialect", "sarcasm"]]

iSarcasmEvalTrain = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/train/train.Ar.csv")[["text","dialect", "sarcastic"]]
iSarcasmEvalTrain_rephrase = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/train/train.Ar.csv")[["rephrase", "dialect", "sarcastic"]]
iSarcasmEvalTestA = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/test/task_A_Ar_test.csv")[["text", "dialect", "sarcastic"]]
iSarcasmEvalTestC = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/test/task_C_Ar_test.csv")[["text_0", "text_1", "dialect", "sarcastic_id"]]

iSarcasmEvalTrain.columns = ["tweet", "dialect", "sarcasm"]
iSarcasmEvalTrain_rephrase.columns = ["tweet", "dialect", "sarcasm"]
iSarcasmEvalTestA.columns = ["tweet", "dialect", "sarcasm"]

iSarcasmEvalTestC["sarcastic_tweet"] = iSarcasmEvalTestC.apply(lambda row: row["text_1"] if row["sarcastic_id"] == 1 else row["text_0"], axis=1)
iSarcasmEvalTestC["non_sarcastic_tweet"] = iSarcasmEvalTestC.apply(lambda row: row["text_0"] if row["sarcastic_id"] == 1 else row["text_1"], axis=1)
sarcastic_df = pd.DataFrame({"tweet": iSarcasmEvalTestC["sarcastic_tweet"], "dialect": iSarcasmEvalTestC["dialect"], "sarcasm": 1})
non_sarcastic_df = pd.DataFrame({"tweet": iSarcasmEvalTestC["non_sarcastic_tweet"], "dialect": iSarcasmEvalTestC["dialect"], "sarcasm": 0})
iSarcasmEvalTestC = pd.concat([sarcastic_df, non_sarcastic_df], ignore_index=True)

Arsarcasm = pd.concat([ArsarcasmTrain, ArsarcasmTest], ignore_index = True)
iSarcasmEval = pd.concat([iSarcasmEvalTrain, iSarcasmEvalTrain_rephrase, iSarcasmEvalTestA, iSarcasmEvalTestC])
fullDataset = pd.read_csv(r"../../Datasets/full Dataset.csv")[['tweet','sarcasm', "dialect"]]

datasets = {
    "ArsarcasmV2 Dataset": ArsarcasmTrain,
    "iSarcasmEval Dataset": iSarcasmEvalTrain,
    "Full Combined Dataset": fullDataset
}



for datasetName, dataset in datasets.items():
    dataset = preProcessData.preProcessData(dataset.copy(deep=True))

    tweets = dataset['tweet'].tolist()
    labels = dataset['sarcasm']

    features = process_in_batches(tweets, batch_size)
    train_tweet, test_tweet, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Define your pipelines for different models
    pipelines = {
        'Logistic_Regression': make_pipeline(StandardScaler(), LogisticRegression()),
        'Ridge_Classifier': make_pipeline(StandardScaler(), RidgeClassifier()),
        'Lin_Support_Vector_Class': make_pipeline(StandardScaler(), LinearSVC()),
        'KNearest_Neighbors': make_pipeline(StandardScaler(), KNeighborsClassifier()),
        'Naive_Bayes': make_pipeline(StandardScaler(), GaussianNB())
        }

    # Fit the models
    fit_models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(train_tweet, train_labels)
        fit_models[algo] = model

        model_path = os.path.join(models_dir, f'{algo} [{datasetName}].pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    for algo, model in fit_models.items():
        predicted = model.predict(test_tweet)
        accuracy = accuracy_score(test_labels, predicted)
        print(algo,'\n', classification_report(test_labels, predicted))
        saveFig(test_labels, predicted, accuracy, algo, datasetName)