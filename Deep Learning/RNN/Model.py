import os
import sys

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(r"../../syspath")
os.chdir(dname)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set TensorFlow logging level to 2 (ERROR)

import shutil
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import preProcessData #type: ignore
import tensorflow as tf
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

from keras.utils import plot_model
from keras.utils import pad_sequences
from gensim.models import KeyedVectors
from sklearn.utils import class_weight
from keras.callbacks import TensorBoard
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings(action = 'ignore')
callback = TensorBoard(log_dir='logs/', histogram_freq=1)

if os.path.isdir("logs"):
    shutil.rmtree("logs")



# dataset = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/training_data.csv")
# dataset = pd.read_csv(r"../../../Datasets/GPT Dataset.csv")
dataset = pd.read_csv(r"../../Datasets/full Dataset.csv")
# dataset, datasetName = pd.read_csv(r"../../Datasets/balanced.csv"), "balanced dataset from original"

dataset.info()
print(f"\n{dataset.head()}\n")

cleaned_dataset = preProcessData.preProcessData(dataset.copy(deep=True))
print("="*50)
cleaned_dataset.info()
print(f"{cleaned_dataset.head()}")
print("="*50)



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
embeddings_file = r"..\..\Embeddings\Aravec CBOW Model\tweets_cbow_300"
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

print("Embedding Matrix shape:", embedding_matrix.shape, "\n")



model = tf.keras.Sequential([
    # Embedding layer for creating word embeddings
    tf.keras.layers.Embedding(vocab_size, TOTAL_EMBEDDING_DIM, input_length=max_length, trainable=True),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout = 0.5, return_sequences = True)),

    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(4, dropout = 0.5)),

    tf.keras.layers.Dense(1, activation="sigmoid")
])


# compile the model
model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), metrics=["accuracy"])



# summarize the model
print(f"\n{model.summary()}")
print("\n # Wait just Fitting model on training data")
plot_model(model, to_file='summary.png', show_shapes=True, show_layer_names=True, dpi=1000)



# splits into traint, validation, and test
train_tweet, test_tweet, train_labels, test_labels = train_test_split(padded_docs, cleaned_dataset["sarcasm"].to_numpy(), test_size=0.20)
#train_tweet, val_tweet, train_labels, val_labels = train_test_split(train_tweet, train_labels, test_size=0.20)



# sm = SMOTE()
# tweet_train, labeled_train = sm.fit_resample(tweet_train, labeled_train) # type: ignore



# fit the model
class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weights))
result = model.fit(train_tweet, train_labels, epochs = 40, verbose = 1, class_weight=class_weights, callbacks=[callback]) # type: ignore



# get Classification report
print()
predicted = np.round(model.predict(test_tweet))
accuracy = accuracy_score(test_labels, predicted)
report = classification_report(test_labels, predicted, target_names=["non-Sarcasm", "Sarcasm"])

print(f"\n{report}\n")



confusionMatrix = confusion_matrix(test_labels, predicted)

ax= plt.subplot()
sns.heatmap(confusionMatrix, annot=True, fmt='g', ax=ax, cmap="viridis") # annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title(f"Accuracy: {accuracy*100:.2f}%")
ax.xaxis.set_ticklabels(["non-Sarcasm", "Sarcasm"])
ax.yaxis.set_ticklabels(["non-Sarcasm", "Sarcasm"])

plt.savefig(f"keras - Full dataset.png", dpi=1000)
plt.close()



# Plot results
acc = result.history['accuracy']
#val_acc = result.history['val_accuracy']
loss = result.history['loss']
#val_loss = result.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'g', label='Training accuracy')
plt.title('Training accuracy')
plt.legend()
plt.savefig(f"Training vs validation accuracy", dpi=1000)

plt.close()

plt.plot(epochs, loss, 'g', label='Training loss')
#plt.plot(epochs, 'r', label='Validation loss')
plt.title('Training loss')
plt.legend()
plt.savefig(f"Training loss", dpi=1000)

plt.close()