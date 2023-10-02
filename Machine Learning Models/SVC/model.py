import os
import sys

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
#sys.path.append(r"C:\Users\Mohamed\Documents\Fall 2023 - 2024\Senior Project in CS\sysPath")
sys.path.append(r"C:\Users\Perseus\Documents\GitHub\ArabicSarcasmDetection\sysPath")

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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from matplotlib import style

le = LabelEncoder()
style.use("ggplot")
startTime = time.time()

if not os.path.isfile(r"Description.txt"):
    descriptionFile = open(r"Description.txt", "w")
    descriptionFile.close()



# dataset, datasetName = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/training_data.csv"), "Original Dataset"
# dataset, datasetName = pd.read_csv(r"https://raw.githubusercontent.com/YaserHabib/ArabicSarcasmDetection/main/Datasets/GPT%20Dataset.csv?token=GHSAT0AAAAAACHECZVVEDC4VQMJ6AO6XPF6ZI3AVUQ"), "GPT Combined Dataset"
# dataset, datasetName = pd.read_csv(r"https://raw.githubusercontent.com/YaserHabib/ArabicSarcasmDetection/main/Datasets/full%20Dataset.csv?token=GHSAT0AAAAAACHECZVV62FP7QAYRUGTXS5OZI3AWCA"), "Full Combined Dataset"
dataset, datasetName = pd.read_csv(r"https://raw.githubusercontent.com/YaserHabib/ArabicSarcasmDetection/main/Datasets/augmented%20Dataset.csv?token=GHSAT0AAAAAACHECZVUTZ6CBJRGKDJQ2ES6ZI3AWPQ"), "Augmented Combined Dataset"
# dataset, datasetName = pd.read_csv(r"https://raw.githubusercontent.com/YaserHabib/ArabicSarcasmDetection/main/Datasets/backtrans%20Dataset.csv?token=GHSAT0AAAAAACHECZVUEQVE7JMQ6CNN6MZQZI3AW4Q"), "Back Translated Combined Dataset"
# dataset, datasetName = pd.read_csv(r"https://raw.githubusercontent.com/YaserHabib/ArabicSarcasmDetection/main/Datasets/synrep%20Dataset.csv?token=GHSAT0AAAAAACHECZVV4OACFCEJRRPXYNYAZI3AXGQ"), "Synonym Replacement Combined Dataset"
# dataset, datasetName = pd.read_csv(r"https://raw.githubusercontent.com/YaserHabib/ArabicSarcasmDetection/main/Datasets/backGPT%20Dataset.csv?token=GHSAT0AAAAAACHECZVVPUX7QVAOR4CCXT4SZI3AXSA"), "Back Translated & GPT Combined Dataset"
# dataset, datasetName = pd.read_csv(r"https://raw.githubusercontent.com/YaserHabib/ArabicSarcasmDetection/main/Datasets/synGPT%20Dataset.csv?token=GHSAT0AAAAAACHECZVUYCQXKVMGYNNG2DIWZI3AX6A"), "Synonym Replacement & GPT Combined Dataset"

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



# columns = ["A", "B", "C", "D", "E"]
# columns = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
columns = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]
padded_docs = PCA(n_components=len(columns)).fit_transform(padded_docs)
padded_docs = StandardScaler().fit_transform(padded_docs)
padded_docs = MinMaxScaler().fit_transform(padded_docs)

features = pd.DataFrame(padded_docs, columns=columns)
features["sentiment"] = cleaned_dataset[["sentiment"]].copy()
features["dialect"] = cleaned_dataset[["dialect"]].copy()

labels = cleaned_dataset[["sarcasm"]].copy()



# splits into traint & test
tweet_train, tweet_test, labeled_train, labeled_test = train_test_split(features.to_numpy(), labels.to_numpy(), test_size=0.2, shuffle=True)



# fit the model
# kernel = "linear"
# kernel = "poly"
# kernel = "rbf"
kernel = "sigmoid"

svc = SVC(kernel=kernel, C=1.0)
svc.fit(tweet_train, labeled_train)



#evaluate the model
trainScore = svc.score(tweet_train, labeled_train)
testScore = svc.score(tweet_test, labeled_test)
labelPredicted = svc.predict(tweet_test)
endTime = time.time()

print()

print(f"\nSVC model score on the training dataset: {trainScore:.2f}")
print(f"SVC model score on the test dataset:     {testScore:.2f}\n")

print(classification_report(labeled_test, labelPredicted, target_names=["Class: 0", "Class: 1"]))

with open(r"Description.txt", "a") as descriptionFile:
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
plt.show()