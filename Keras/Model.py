import os
import sys

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(r"C:\Users\Mohamed\Documents\Fall 2023 - 2024\Senior Project in CS\sysPath")
os.chdir(dname)

import shutil
import warnings
import numpy as np
import pandas as pd
import preProcessData
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.callbacks import TensorBoard
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model, pad_sequences

from keras.layers import Embedding
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional, GRU

from gensim.models import KeyedVectors
from matplotlib import style

le = LabelEncoder()
style.use("ggplot")

if os.path.isdir("logs"):
    shutil.rmtree("logs")
callback = TensorBoard(log_dir=rf'logs/', histogram_freq=1)



dataset = pd.read_csv(r"C:\Users\Mohamed\Documents\Fall 2023 - 2024\Senior Project in CS\Total Dataset.csv")
dataset.info()
# display the dataset before the pre-processing
print(f"\n{dataset.head()}")



cleaned_dataset = preProcessData.preProcessData(dataset.copy(deep=True))
# cleaned_dataset.to_excel('cleanedset.xlsx', index=False)



# prepare tokenizer
T = Tokenizer()
T.fit_on_texts(cleaned_dataset["tweet"].tolist())
vocab_size = len(T.word_index) + 1



# integer encode the documents
encoded_docs = T.texts_to_sequences(cleaned_dataset["tweet"].tolist())
# print("encoded_docs:\n",encoded_docs)



# pad documents to a max length of 4 words
max_length = len(max(np.array(cleaned_dataset["tweet"]), key=len))
padded_docs = pad_sequences(encoded_docs, maxlen = max_length, padding = "post")
print("\npadded_docs:\n",padded_docs)



# load the whole embedding into memory
w2v_embeddings_index = {}
TOTAL_EMBEDDING_DIM = 300
# Aravec Twitter-CBOW Model ===> https://bakrianoo.ewr1.vultrobjects.com/aravec/full_grams_cbow_300_twitter.zip
# Aravec Twitter-SkipGram Model ===> https://bakrianoo.ewr1.vultrobjects.com/aravec/full_grams_sg_300_twitter.zip
# Aravec Wikipedia-CBOW Model ===> https://bakrianoo.ewr1.vultrobjects.com/aravec/full_grams_cbow_300_wiki.zip
# Aravec Wikipedia-SkipGram Model ===> https://bakrianoo.ewr1.vultrobjects.com/aravec/full_grams_sg_300_wiki.zip
embeddings_file = r"C:\Users\Mohamed\Documents\Fall 2023 - 2024\Senior Project in CS\full_grams_cbow_300_twitter\full_grams_cbow_300_twitter.mdl"
w2v_model = KeyedVectors.load(embeddings_file)

for word in w2v_model.wv.index_to_key:
    w2v_embeddings_index[word] = w2v_model.wv[word]

print("\nLoaded %s word vectors."% len(w2v_embeddings_index))



# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, TOTAL_EMBEDDING_DIM))

for word, i in T.word_index.items():
    embedding_vector = w2v_embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print("\nEmbedding Matrix shape:", embedding_matrix.shape)



# define model
# embedding_layer = tf.keras.layers.Embedding(vocab_size, TOTAL_EMBEDDING_DIM, weights=[embedding_matrix], input_length=4, trainable=False)
# input_placeholder= tf.keras.Input(shape=(max_length,), dtype = "int32")
# input_embedding = embedding_layer(input_placeholder)
# drop = tf.keras.layers.Dropout(0.4)
# lstm = tf.keras.layers.LSTM(units = 10, activation = "relu")(input_embedding)
# drop = tf.keras.layers.Dropout(0.6)
# preds = tf.keras.layers.Dense(1, activation = "sigmoid", name = "activation")(lstm)
# model = tf.keras.models.Model(inputs = input_placeholder, outputs = preds)

#Defining Neural Network
model = Sequential()
#Non-trainable embeddidng layer
model.add(Embedding(vocab_size, output_dim=TOTAL_EMBEDDING_DIM, weights=[embedding_matrix], input_length=508, trainable=True))
#LSTM 
model.add(Bidirectional(LSTM(units=128 , recurrent_dropout = 0.3 , dropout = 0.3,return_sequences = True)))
model.add(Bidirectional(GRU(units=32 , recurrent_dropout = 0.1 , dropout = 0.1)))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999), metrics=["accuracy"])



# summarize the model
print(f"\n{model.summary()}")
print("\n # Wait just Fitting model on training data")
plot_model(model, to_file='summary.png', show_shapes=True, show_layer_names=True)



labels = cleaned_dataset[["sarcasm"]].copy()



# splits into traint & test
tweet_train, tweet_test, labeled_train, labeled_test = train_test_split(padded_docs, labels.to_numpy(), test_size=0.2, shuffle=True)



# fit the model
model.fit(tweet_train, labeled_train, epochs = 2, verbose = 1, callbacks=[callback]) # type: ignore



#evaluate the model
print(model.predict(tweet_test))

trainLoss, trainScore = model.evaluate(tweet_train, labeled_train)
testLoss, testScore = model.evaluate(tweet_test, labeled_test)
labelPredicted = np.round(model.predict(tweet_test))

print()

print(f"keras score on the training dataset: {trainScore:.2f}")
print(f"keras score on the test dataset:     {testScore:.2f}\n")

print(classification_report(labeled_test, labelPredicted, target_names=["Class: 0", "Class: 1"]))



confusionMatrix = confusion_matrix(labeled_test, labelPredicted)

# confMatrix_display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=[0, 1])
ax= plt.subplot()
sns.heatmap(confusionMatrix, annot=True, fmt='g', ax=ax, cmap="viridis") # annot=True to annotate cells, ftm='g' to disable scientific notation
# sns.heatmap(confusionMatrix, annot=True, fmt='g', ax=ax, cmap="Blues") # annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title(f"Accuracy: {testScore*100:.2f}%")

ax.xaxis.set_ticklabels([0, 1])
ax.yaxis.set_ticklabels([0, 1])

plt.savefig(r"keras confusionMatrix.png", dpi=1000)
plt.show()