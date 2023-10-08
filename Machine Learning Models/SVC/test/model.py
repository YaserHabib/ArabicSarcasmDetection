import os
import sys

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(r"C:\Users\Mohamed\Documents\Fall 2023 - 2024\Senior Project in CS\sysPath")
os.chdir(dname)

import time
import numpy as np
import pandas as pd
import seaborn as sns
import preProcessData #type: ignore
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

from matplotlib import style

le = LabelEncoder()
style.use("ggplot")
startTime = time.time()

if not os.path.isfile(r"Description.txt"):
    descriptionFile = open(r"Description.txt", "w")
    descriptionFile.close()



def Plotting(datasetName, kernel, features, testScore, labeled_test, labelPredicted):
    confusionMatrix = confusion_matrix(labeled_test, labelPredicted)

    ax= plt.subplot()
    sns.heatmap(confusionMatrix, annot=True, fmt='g', ax=ax, cmap="viridis") # annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f"Accuracy: {testScore*100:.2f}%")

    ax.xaxis.set_ticklabels(["0", "1"])
    ax.yaxis.set_ticklabels(["0", "1"])

    plt.savefig(f"{datasetName} - {kernel} kernel - {len(features.columns)}DR.png", dpi=1000)



def Save(kernel, features, datasetName, trainScore, testScore, labeled_test, labelPredicted):

    with open(r"Description.txt", "a") as descriptionFile:
        endTime = time.time()

        descriptionFile.write("="*100)
        descriptionFile.write(f"\nKernel used:  {kernel}")
        descriptionFile.write(f"\nDimention Reduction:  {len(features.columns)}")
        descriptionFile.write(f"\nDataset used: {datasetName}\n")
        descriptionFile.write(f"\nSVC model score on the training dataset:  {trainScore:.2f}")
        descriptionFile.write(f"\nSVC model score on the test dataset:      {testScore:.2f}")
        descriptionFile.write(f"\nExecution Time:   {endTime - startTime}s\n\n")
        descriptionFile.write(classification_report(labeled_test, labelPredicted, target_names=["Class: 0", "Class: 1"])) # type: ignore
        descriptionFile.write("\n\n")
        descriptionFile.close()



def fit(Classifier, kernel, tweet_train, labeled_train, tweet_test, labeled_test):
    classifier = Classifier(kernel=kernel)
    classifier.fit(tweet_train, labeled_train)

    trainScore = classifier.score(tweet_train, labeled_train)
    testScore = classifier.score(tweet_test, labeled_test)
    labelPredicted = classifier.predict(tweet_test)

    return trainScore, testScore, labelPredicted



def Feature_preparing(cleaned_dataset, padded_docs, columns):

    padded_docs = PCA(n_components=len(columns)).fit_transform(padded_docs)
    padded_docs = StandardScaler().fit_transform(padded_docs)
    padded_docs = MinMaxScaler().fit_transform(padded_docs)

    features = pd.DataFrame(padded_docs, columns=columns)
    features["sentiment"] = cleaned_dataset[["sentiment"]].copy()
    features["dialect"] = cleaned_dataset[["dialect"]].copy()

    labels = cleaned_dataset[["sarcasm"]].copy()

    return features, labels



def Training(cleaned_dataset, datasetName, padded_docs, columns, kernels):
    result = pd.DataFrame(columns=["Dimension Reduction", "Kernel", "Precision Score", "Recall Score", "F1-Score", "Accuracy"])

    for column in columns.values():
        for kernel in kernels.values():

            features, labels = Feature_preparing(cleaned_dataset, padded_docs, column)
            tweet_train, tweet_test, labeled_train, labeled_test = train_test_split(features.to_numpy(), labels.to_numpy(), test_size=0.2, shuffle=True)

            trainScore, testScore, labelPredicted = fit(SVC, kernel, tweet_train, labeled_train, tweet_test, labeled_test)

            classificationReport = classification_report(labeled_test, labelPredicted, target_names=["Class: 0", "Class: 1"])
            print(classificationReport)

            Save(kernel, features, datasetName, trainScore, testScore, labeled_test, labelPredicted)
            Plotting(datasetName, kernel, features, testScore, labeled_test, labelPredicted)

            precision = precision_score(labeled_test, labelPredicted)
            recall = recall_score(labeled_test, labelPredicted)
            f1 = f1_score(labeled_test, labelPredicted)

            result.loc[len(result.index)] = [len(features.columns), kernel, precision, recall, f1, testScore] # type: ignore

    result.to_excel("Test Result.xlsx", index=False)



# dataset, datasetName = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/training_data.csv"), "Original Dataset"
dataset, datasetName = pd.read_csv(r"../../../Datasets/GPT Dataset.csv"), "GPT Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../Datasets/full Dataset.csv"), "Full Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../Datasets/augmented Dataset.csv"), "Augmented Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../Datasets/backtrans Dataset.csv"), "Back Translated Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../Datasets/synrep Dataset.csv"), "Synonym Replacement Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../Datasets/backGPT Dataset.csv"), "Back Translated & GPT Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../Datasets/synGPT Dataset.csv"), "Synonym Replacement & GPT Combined Dataset"

dataset.info()
print(f"\n{dataset.head()}")



cleaned_dataset = preProcessData.preProcessData(dataset.copy(deep=True))
# cleaned_dataset.to_excel('cleanedset.xlsx', index=False)



# prepare tokenizer
T = Tokenizer()
T.fit_on_texts(cleaned_dataset["tweet"].tolist())
vocab_size = len(T.word_index) + 1



# integer encode the documents
encoded_docs = T.texts_to_sequences(cleaned_dataset["tweet"].tolist())



# pad documents to a max length of 4 words
max_length = len(max(np.array(cleaned_dataset["tweet"]), key=len))
padded_docs = pad_sequences(encoded_docs, maxlen = max_length, padding = "post")
print("\npadded_docs:\n\n",padded_docs)



columnsA = ["A", "B", "C", "D", "E"]
columnsB = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
columnsC = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]
columns = {"columnsA":columnsA, "columnsB": columnsB, "columnsC": columnsC}



kernels = {"Linear": "linear", "Poly": "poly", "RBF": "rbf", "Sigmoid": "sigmoid"}



Training(cleaned_dataset, datasetName, padded_docs, columns, kernels)