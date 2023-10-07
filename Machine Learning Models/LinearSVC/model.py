import os
import sys

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(r"C:\Users\Admin\OneDrive\Desktop\Project\ArabicSarcasmDetection\sysPath")
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
# dataset, datasetName = pd.read_csv(r"../../../Datasets/GPT Dataset.csv"), "GPT Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../../Datasets/full Dataset.csv"), "Full Combined Dataset"
dataset, datasetName = pd.read_csv(r"../../Datasets/augmented Dataset.csv"), "Augmented Combined Dataset"
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
print("\npadded_docs:\n\n",padded_docs)



columns = ["A", "B", "C", "D", "E"]
# columns = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
# columns = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]
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
# The combination of penalty='l1' and loss='hinge' is not supported.

penalty, loss, dual, max_iter = "l1", "squared_hinge", False, 100
# penalty, loss, dual, max_iter = "l1", "squared_hinge", False, 1000
# penalty, loss, dual, max_iter = "l1", "squared_hinge", False, 10000

# penalty, loss, dual, max_iter = "l2", "squared_hinge", True, 100
# penalty, loss, dual, max_iter = "l2", "squared_hinge", True, 1000
# penalty, loss, dual, max_iter = "l2", "squared_hinge", True, 10000

# penalty, loss, dual, max_iter = "l2", "hinge", True, 100
# penalty, loss, dual, max_iter = "l2", "hinge", True, 1000
# penalty, loss, dual, max_iter = "l2", "hinge", True, 10000

svc = LinearSVC(penalty=penalty, loss=loss, dual=dual, C=1.0, max_iter=max_iter)
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
    descriptionFile.write(f"\ndual:                 {dual}")
    descriptionFile.write(f"\nmax_iter:             {max_iter}")
    descriptionFile.write(f"\nloss used:            {loss}")
    descriptionFile.write(f"\nPenalty used:         {penalty}")
    descriptionFile.write(f"\nDimention Reduction:  {len(features.columns)}\n")
    descriptionFile.write(f"\nDataset used:         {datasetName}\n")
    descriptionFile.write(f"\nSVC model score on the training dataset:  {trainScore:.2f}")
    descriptionFile.write(f"\nSVC model score on the test dataset:      {testScore:.2f}")
    descriptionFile.write(f"\nExecution Time:       {endTime - startTime}s\n\n")
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

plt.savefig(f"{datasetName} - {penalty} Penalty - {loss} Loss - {max_iter} max_iter - {len(features.columns)}DR.png", dpi=1000)
plt.show()