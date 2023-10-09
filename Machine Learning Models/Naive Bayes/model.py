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

from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer

from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
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



def featureEngineering(dataset, padded_docs, columns):
    labels = dataset["sarcasm"]
    dataset = dataset.drop(["sarcasm"], axis=1)

    padded_docs = PCA(n_components=len(columns)).fit_transform(padded_docs)
    padded_docs = StandardScaler().fit_transform(padded_docs)
    padded_docs = MinMaxScaler().fit_transform(padded_docs)

    features = pd.DataFrame(padded_docs, columns=columns)
    for column in dataset.columns:
        features[column] = dataset[column]

    return features, labels



def modelEvaluation(classifier, train, test):
    trainScore = classifier.score(train["data"], train["labels"])
    testScore = classifier.score(test["data"], test["labels"])
    labelPredicted = classifier.predict(test["data"])

    return trainScore, testScore, labelPredicted


def display(dimentionReduction, datasetName, trainScore, testScore, classificationReport, startTime):
    endTime = time.time()
    
    print("="*100)
    print(f"Dimention Reduction:  {dimentionReduction}")
    print(f"Dataset used: {datasetName}")
    print(f"\nSVC model score on the training dataset:  {trainScore:.2f}")
    print(f"SVC model score on the test dataset:      {testScore:.2f}")
    print(f"Execution Time:   {abs(endTime - startTime)}s\n")
    print(classificationReport) # type: ignore
    print("\n")



def recordResult(dimentionReduction, datasetName, trainScore, testScore, classificationReport, startTime):
    endTime = time.time()
    
    with open(r"Description.txt", "a") as descriptionFile:
        descriptionFile.write("="*100)
        descriptionFile.close()

    with open(r"Description.txt", "a") as descriptionFile:
        descriptionFile.write(f"\nDimention Reduction:  {dimentionReduction}")
        descriptionFile.write(f"\nDataset used: {datasetName}\n")
        descriptionFile.write(f"\nSVC model score on the training dataset:  {trainScore:.2f}")
        descriptionFile.write(f"\nSVC model score on the test dataset:      {testScore:.2f}")
        descriptionFile.write(f"\nExecution Time:   {abs(endTime - startTime)}s\n\n")
        descriptionFile.write(classificationReport) # type: ignore
        descriptionFile.write("\n\n")
        descriptionFile.close()



def saveFig(datasetName, DR, true, predicted, testScore):
    confusionMatrix = confusion_matrix(true, predicted)
    ax= plt.subplot()
    sns.heatmap(confusionMatrix, annot=True, fmt='g', ax=ax, cmap="viridis") # annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f"Accuracy: {testScore*100:.2f}%")

    ax.xaxis.set_ticklabels(["0", "1"])
    ax.yaxis.set_ticklabels(["0", "1"])

    plt.savefig(f"{datasetName} - {len(DR)}DR.png", dpi=1000)
    plt.close()



def recordXLSX(data, DR, precision, recall, f1, testScore):
    data.loc[len(data.index)] = [int(DR), precision, recall, f1, testScore] # type: ignore



def barBlot(datasetName, data):    
    names = data["DR"].tolist()

    f1 = data["F1-score"]
    accuracy = data["Accuracy"]
    measures = pd.concat([f1, accuracy], axis=1, join='inner')
    
    x = np.arange(len(names))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize =(15, 10))

    for attribute, measurement in measures.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("percentage (%)")
    ax.set_xlabel(f"Variables: Dimention Reduction")
    ax.set_title('visualize the Impact of variables')

    ax.set_xticks(x + (width/2), names)

    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(0, 1)

    plt.savefig(f"{datasetName}DR.png", dpi=1000)
    plt.close()



def MLprocess(dataset, padded_docs, columns, datasetName, Ctime):
    dataset = dataset.drop(["tweet"], axis=1)
    data = pd.DataFrame(columns=["DR", "Precision-score", "Recall-score", "F1-score", "Accuracy"])
    startTime = time.time() - Ctime

    for column in columns:
        preTime = time.time() + startTime

        features, labels = featureEngineering(dataset, padded_docs, columns[column])
        tweet_train, tweet_test, labeled_train, labeled_test = train_test_split(features.to_numpy(), labels.to_numpy(), test_size=0.2, shuffle=True)
        
        naive_bayes = GaussianNB()
        naive_bayes.fit(tweet_train, labeled_train)
        test = {"data": tweet_test, "labels": labeled_test}
        train = {"data": tweet_train, "labels": labeled_train}

        trainScore, testScore, labelPredicted = modelEvaluation(naive_bayes, train, test)
    
        classificationReport = classification_report(labeled_test, labelPredicted, target_names=["Class: 0", "Class: 1"])

        display(len(features.columns), datasetName, trainScore, testScore, classificationReport, preTime)
        recordResult(len(features.columns), datasetName, trainScore, testScore, classificationReport, preTime)
        saveFig(datasetName, features.columns, labeled_test, labelPredicted, testScore)

        f1 = f1_score(labeled_test, labelPredicted)
        recall = recall_score(labeled_test, labelPredicted)
        precision = precision_score(labeled_test, labelPredicted)
        recordXLSX(data, len(features.columns), precision, recall, f1, testScore)

    barBlot(datasetName, data.copy(deep=True))

    data.to_excel("record.xlsx", index=False)



dataset, datasetName = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/training_data.csv"), "Original Dataset"
# dataset, datasetName = pd.read_csv(r"../../../Datasets/GPT Dataset.csv"), "GPT Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../../Datasets/full Dataset.csv"), "Full Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../../Datasets/augmented Dataset.csv"), "Augmented Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../../Datasets/backtrans Dataset.csv"), "Back Translated Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../../Datasets/synrep Dataset.csv"), "Synonym Replacement Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../../Datasets/backGPT Dataset.csv"), "Back Translated & GPT Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../../Datasets/synGPT Dataset.csv"), "Synonym Replacement & GPT Combined Dataset"

dataset.info()
print(f"\n{dataset.head()}")



cleaned_dataset = preProcessData.preProcessData(dataset.copy(deep=True))



# prepare tokenizer
T = Tokenizer()
T.fit_on_texts(cleaned_dataset["tweet"].tolist())
vocab_size = len(T.word_index) + 1



# integer encode the documents
encoded_docs = T.texts_to_sequences(cleaned_dataset["tweet"].tolist())



# pad documents to a max length of 4 words
max_length = len(max(np.array(cleaned_dataset["tweet"]), key=len))
padded_docs = pad_sequences(encoded_docs, maxlen = max_length, padding = "post")
print("\npadded_docs:\n\n", padded_docs, "\n")



columnsA = ["A", "B", "C", "D", "E"]
columnsB = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
columnsC = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]
columns = {"columnsA": columnsA, "columnsB": columnsB, "columnsC": columnsC}

MLprocess(cleaned_dataset, padded_docs, columns, datasetName, startTime)