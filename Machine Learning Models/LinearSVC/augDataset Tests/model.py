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

from sklearn.svm import LinearSVC
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



def display(variables, dimentionReduction, datasetName, trainScore, testScore, classificationReport, vars, startTime):
    endTime = time.time()
    vars.sort()

    print("="*100)
    for var in vars: print(f"{var} used:\t{variables[var]}")
    print(f"Dimention Reduction:  {dimentionReduction}")
    print(f"Dataset used: {datasetName}")
    print(f"\nSVC model score on the training dataset:  {trainScore:.2f}")
    print(f"SVC model score on the test dataset:      {testScore:.2f}")
    print(f"Execution Time:   {abs(endTime - startTime)}s\n")
    print(classificationReport) # type: ignore
    print("\n")




def recordResult(variables, dimentionReduction, datasetName, trainScore, testScore, classificationReport, vars, startTime):
    endTime = time.time()
    vars.sort()
    
    with open(r"Description.txt", "a") as descriptionFile:
        descriptionFile.write("="*100)
    descriptionFile.close()

    with open(r"Description.txt", "a") as descriptionFile:
        for var in vars:
            descriptionFile.write(f"\n{var} used:\t{variables[var]}")
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



def saveFig(datasetName, variables, DR, true, predicted, testScore):
    confusionMatrix = confusion_matrix(true, predicted)
    ax= plt.subplot()
    sns.heatmap(confusionMatrix, annot=True, fmt='g', ax=ax, cmap="viridis") # annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f"Accuracy: {testScore*100:.2f}%")

    ax.xaxis.set_ticklabels(["0", "1"])
    ax.yaxis.set_ticklabels(["0", "1"])
    variables = str(variables).replace('\'', '').replace(':', '=').replace(', ', " - ")
    
    plt.savefig(f"{datasetName} - {variables} - {len(DR)}DR.png", dpi=1000)
    plt.close()


def recordXLSX(data, DR, classifierVariables, precision, recall, f1, testScore):
    penalty = classifierVariables["penalty"]; loss = classifierVariables["loss"]
    dual = classifierVariables["dual"]; max_iter = classifierVariables["max_iter"]

    data.loc[len(data.index)] = [DR, penalty, loss, dual, max_iter, precision, recall, f1, testScore] # type: ignore



def barBlot(DR, data):
    variables = ["penalty", "loss", "max_iter"]

    loss = data[data.DR == DR]["loss"].tolist()
    penalty = data[data.DR==DR]["penalty"].tolist()
    max_iter = data[data.DR == DR]["max_iter"].tolist()

    names = list()
    for index in range(len(loss)):
        names.append(penalty[index] + "\n" + loss[index] + "\n" + str(max_iter[index]))

    f1 = data[data.DR == DR]["F1-score"]
    accuracy = data[data.DR == DR]["Accuracy"]
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
    ax.set_xlabel(f"Variables: {variables}")
    ax.set_title("visualize the Impact of variables")

    ax.set_xticks(x + (width/2), names)

    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(0, 1)
    
    plt.savefig(f"{DR}DR.png", dpi=1000)
    plt.close()



def MLprocess(dataset, padded_docs, columns, classifierVariables, datasetName, vars, Ctime):
    dataset = dataset.drop(["tweet"], axis=1)
    data = pd.DataFrame(columns=["DR", "penalty", "loss", "dual", "max_iter", "Precision-score", "Recall-score", "F1-score", "Accuracy"])
    startTime = time.time() - Ctime

    for column in columns:
        features, labels = featureEngineering(dataset, padded_docs, columns[column])
        tweet_train, tweet_test, labeled_train, labeled_test = train_test_split(features.to_numpy(), labels.to_numpy(), test_size=0.2, shuffle=True)

        for row in classifierVariables:
            preTime = time.time() + startTime

            loss = classifierVariables[row]["loss"]
            dual = classifierVariables[row]["dual"]
            penalty = classifierVariables[row]["penalty"]
            max_iter = classifierVariables[row]["max_iter"]

            svc = LinearSVC(penalty=penalty, loss=loss, dual=dual, max_iter=max_iter)
            svc.fit(tweet_train, labeled_train)

            test = {"data": tweet_test, "labels": labeled_test}
            train = {"data": tweet_train, "labels": labeled_train}

            trainScore, testScore, labelPredicted = modelEvaluation(svc, train, test)
            classifierVars = {"penalty": penalty, "loss": loss, "dual": dual, "max_iter": max_iter}
            classificationReport = classification_report(labeled_test, labelPredicted, target_names=["Class: 0", "Class: 1"])

            display(classifierVars, len(features.columns), datasetName, trainScore, testScore, classificationReport, vars, preTime)
            recordResult(classifierVars, len(features.columns), datasetName, trainScore, testScore, classificationReport, vars, preTime)
            saveFig(datasetName, classifierVars, features.columns, labeled_test, labelPredicted, testScore)

            f1 = f1_score(labeled_test, labelPredicted)
            recall = recall_score(labeled_test, labelPredicted)
            precision = precision_score(labeled_test, labelPredicted)
            recordXLSX(data, len(features.columns), classifierVars, precision, recall, f1, testScore)

        barBlot(len(features.columns), data.copy(deep=True))

    data.to_excel("record.xlsx", index=False)



# dataset, datasetName = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/training_data.csv"), "Original Dataset"
# dataset, datasetName = pd.read_csv(r"../../../Datasets/GPT Dataset.csv"), "GPT Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../../Datasets/full Dataset.csv"), "Full Combined Dataset"
dataset, datasetName = pd.read_csv(r"../../../Datasets/augmented Dataset.csv"), "Augmented Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../../Datasets/backtrans Dataset.csv"), "Back Translated Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../../Datasets/synrep Dataset.csv"), "Synonym Replacement Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../../Datasets/backGPT Dataset.csv"), "Back Translated & GPT Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../../Datasets/synGPT Dataset.csv"), "Synonym Replacement & GPT Combined Dataset"

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
print("\npadded_docs:\n\n", padded_docs, "\n")



columnsA = ["A", "B", "C", "D", "E"]
columnsB = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
columnsC = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]
columns = {"columnsA": columnsA, "columnsB": columnsB, "columnsC": columnsC}



# fit the model
classifierVariables = {
    "firstRow": {"penalty": "l1", "loss": "squared_hinge", "dual": False, "max_iter": 100},
    "secondRow": {"penalty": "l1", "loss": "squared_hinge", "dual": False, "max_iter": 1000},
    "thirdRow": {"penalty": "l1", "loss": "squared_hinge", "dual": False, "max_iter": 10000},

    "fourthRow": {"penalty": "l2", "loss": "squared_hinge", "dual": True, "max_iter": 100},
    "fifthRow": {"penalty": "l2", "loss": "squared_hinge", "dual": True, "max_iter": 1000},
    "sixthRow": {"penalty": "l2", "loss": "squared_hinge", "dual": True, "max_iter": 10000},

    "seventhRow": {"penalty": "l2", "loss": "hinge", "dual": True, "max_iter": 100},
    "eighthRow": {"penalty": "l2", "loss": "hinge", "dual": True, "max_iter": 1000},
    "ninethRow": {"penalty": "l2", "loss": "hinge", "dual": True, "max_iter": 10000},
    }
variables = ["penalty", "loss", "dual", "max_iter"]

MLprocess(cleaned_dataset, padded_docs, columns, classifierVariables, datasetName, variables, startTime)