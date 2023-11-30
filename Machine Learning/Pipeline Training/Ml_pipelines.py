import os
import sys

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(r"..\..\sysPath")
os.chdir(dname)

models_dir = os.path.join(dname, 'Trained Models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

import preProcessData  #type: ignore
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report # Accuracy metrics 
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


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

dataset = pd.read_csv(r"../../Datasets/full Dataset.csv")
dataset = preProcessData.preProcessData(dataset.copy(deep=True))
tweets = dataset['tweet'].tolist()
print(len(tweets))
labels = dataset['sarcasm']

features = process_in_batches(tweets, batch_size)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

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
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

    model_path = os.path.join(models_dir, f'{algo}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo,'\n', classification_report(y_test, yhat))