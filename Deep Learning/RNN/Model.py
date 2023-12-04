import os
import sys
import time

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(r"../../sysPath")
os.chdir(dname)

models_dir = os.path.join(dname, 'Trained Models [RNN]')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set TensorFlow logging level to 2 (ERROR)

import shutil
import warnings
import pandas as pd
import tensorflow as tf

from summarize import prepareData, summarize, TrainTestSplit, smote, fit, modelEvaluation, display # type: ignore
from summarize import recordResult, plotPerformance, saveFig, recordXLSX, barPlot # type: ignore
from keras.callbacks import TensorBoard

warnings.filterwarnings(action = 'ignore')
callback = TensorBoard(log_dir = 'logs/', histogram_freq = 1)

if os.path.isdir("logs"):
    shutil.rmtree("logs")



def configModel(max_length, vocab_size, TOTAL_EMBEDDING_DIM, embedding_matrix):
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

        # First Dense layer with 64 neurons and ReLU activation
        tf.keras.layers.Dense(64, activation = 'relu'),

        # Dropout layer to prevent overfitting
        tf.keras.layers.Dropout(0.5),

        # Final Dense layer with 1 neuron and sigmoid activation for binary classification
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001), metrics = ["accuracy"], loss = "binary_crossentropy")

    return model



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
iSarcasmEval = pd.concat([iSarcasmEvalTrain, iSarcasmEvalTrain_rephrase, iSarcasmEvalTestA, iSarcasmEvalTestC], ignore_index = True)
ARISarcasm = pd.concat([Arsarcasm, iSarcasmEval], ignore_index = True)[["tweet","dialect", "sarcasm"]]
fullDataset = pd.read_csv(r"../../Datasets/full Dataset.csv")[["tweet","dialect", "sarcasm"]]

datasets = {
    "ArsarcasmV2 Dataset": ArsarcasmTrain,
    "iSarcasmEval Dataset": iSarcasmEvalTrain,
    "ARISarcasm": ARISarcasm,
    "Full Combined Dataset": fullDataset
}

data = pd.DataFrame(columns=["Dataset Name", "# non-Sarcasm", "# Sarcasm", "SMOTE state",  "sarcasm:nonsarcasm", "Precision-Score", "Recall-Score", "F1-Score", "Accuracy"])
smoteControl = False



for datasetName, dataset in datasets.items():
    start = time.time()
    ratio = dataset["sarcasm"].value_counts()[1] / len(dataset)
    model_path = os.path.join(models_dir, f"{datasetName}.pkl")

    smoteStatus = True if ( 0.40 > ratio or ratio > 0.60) and smoteControl else False

    dataset, max_length, vocab_size, TOTAL_EMBEDDING_DIM, embedding_matrix, padded_docs = prepareData(dataset)

    model = configModel(max_length, vocab_size, TOTAL_EMBEDDING_DIM, embedding_matrix)

    summarize(model, datasetName)

    train_tweet, test_tweet, train_labels, test_labels, val_tweet, val_labels = TrainTestSplit(padded_docs, dataset)

    train_tweet, train_labels = smote(train_tweet, train_labels) if smoteStatus else (train_tweet, train_labels)

    result = fit(model, train_labels, train_tweet, val_tweet, val_labels, 10)

    predicted, precision, accuracy, recall, f1, classificationReport = modelEvaluation(model, test_tweet, test_labels)

    model.save(rf"{models_dir}/{datasetName}.keras")

    end = time.time()

    display(datasetName, classificationReport, ratio, smoteStatus, end-start)

    recordResult(datasetName, classificationReport, ratio, smoteStatus, end-start)

    plotPerformance(result, datasetName)

    saveFig(test_labels, predicted, accuracy, "RNN", datasetName)

    nonSarcasmCount = dataset["sarcasm"].value_counts()[0]
    sarcasmCount = dataset["sarcasm"].value_counts()[1]
    recordXLSX(data, datasetName, nonSarcasmCount, sarcasmCount, ratio, precision, recall, f1, accuracy, smoteStatus)


barPlot(data, "RNN")
recordName = "model performance - SMOTE ON.xlsx" if smoteControl else "model performance.xlsx"
data.to_excel(recordName, index = False)