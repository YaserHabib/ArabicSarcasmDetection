import os
import sys

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(r"../../sysPath")
os.chdir(dname)

models_dir = os.path.join(dname, 'Trained Models [RNN]')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set TensorFlow logging level to 2 (ERROR)

import shutil
import pickle
import warnings
import numpy as np
import pandas as pd
import preProcessData #type: ignore
import tensorflow as tf

from summarize import prepareData, TrainTestSplit, saveFig # type: ignore
from keras.callbacks import TensorBoard

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import classification_report, accuracy_score



warnings.filterwarnings(action = 'ignore')
callback = TensorBoard(log_dir = 'logs/', histogram_freq = 1)

if os.path.isdir("logs"):
    shutil.rmtree("logs")



def extract_CNN_features(max_length, vocab_size, TOTAL_EMBEDDING_DIM, embedding_matrix, padded_docs):
    model = tf.keras.Sequential([
    # Embedding layer for creating word embeddings
    tf.keras.layers.Embedding(vocab_size, TOTAL_EMBEDDING_DIM, weights = [embedding_matrix], input_length = max_length, trainable = True),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout = 0.5, return_sequences = True)),

    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, dropout = 0.5, return_sequences = True)),

    # Conv1D layer for pattern recognition model and extract the feature from the vectors
    tf.keras.layers.Conv1D(filters = 64, kernel_size = 3, activation = "relu"),

    tf.keras.layers.BatchNormalization(),

    # GlobalMaxPooling layer to extract relevant features
    tf.keras.layers.GlobalMaxPool1D(),

    # First Dense layer with 4 neurons and ReLU activation
    tf.keras.layers.Dense(64, activation = 'relu'),

    # Dropout layer to prevent overfitting
    tf.keras.layers.Dropout(0.5),

    # Final Dense layer with 1 neuron and sigmoid activation for binary classification
    tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    model = model.predict(padded_docs)
    print(f"\nmodel shape after feature extraction: {model.shape}\n")

    return model



def getFiture(dataset):
    dataset, max_length, vocab_size, TOTAL_EMBEDDING_DIM, embedding_matrix, padded_docs = prepareData(dataset.copy(deep=True))
    features = extract_CNN_features(max_length, vocab_size, TOTAL_EMBEDDING_DIM, embedding_matrix, padded_docs)

    return features, dataset



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

    features, dataset = getFiture(dataset.copy(deep=True))
    train_tweet, test_tweet, train_labels, test_labels = TrainTestSplit(features, dataset, validation=False)

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