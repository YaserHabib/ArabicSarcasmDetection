import math
import os
import sys

from matplotlib.pylab import beta

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(r"..\..\sysPath")
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
from keras.callbacks import TensorBoard
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings(action = 'ignore')
callback = TensorBoard(log_dir='logs/', histogram_freq=1)

if os.path.isdir("logs"):
    shutil.rmtree("logs")



# dataset, datasetName = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/training_data.csv"), "Original Dataset"
# dataset, datasetName = pd.read_csv(r"../../Datasets/GPT Dataset.csv"), "GPT Combined Dataset"
dataset, datasetName = pd.read_csv(r"../../Datasets/full Dataset.csv"), "Full Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../Datasets/augmented Dataset.csv"), "Augmented Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../Datasets/backtrans Dataset.csv"), "Back Translated Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../Datasets/synrep Dataset.csv"), "Synonym Replacement Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../Datasets/backGPT Dataset.csv"), "Back Translated & GPT Combined Dataset"
# dataset, datasetName = pd.read_csv(r"../../Datasets/synGPT Dataset.csv"), "Synonym Replacement & GPT Combined Dataset"

dataset.info()
print(f"\n{dataset.head()}")

cleaned_dataset = preProcessData.preProcessData(dataset.copy(deep=True))



# prepare tokenizer
T = Tokenizer()
T.fit_on_texts(cleaned_dataset["tweet"].tolist())
vocab_size = len(T.word_index) + 1



# integer encode the documents
encoded_docs = T.texts_to_sequences(cleaned_dataset["tweet"].tolist())
# print("encoded_docs:\n",encoded_docs)



# pad documents to a max length of 4 words
max_length = len(max(np.array(dataset["tweet"]), key=len))
padded_docs = pad_sequences(encoded_docs, maxlen = max_length, padding = "post")
print("\npadded_docs:\n",padded_docs)



# load the whole embedding into memory
w2v_embeddings_index = {}
TOTAL_EMBEDDING_DIM = 300
embeddings_file = r"../../Embeddings\Aravec CBOW Model\full_grams_cbow_300_twitter.mdl"
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



# Original

# model = tf.keras.Sequential([

#     # Embedding layer for creating word embeddings
#     tf.keras.layers.Embedding(vocab_size, TOTAL_EMBEDDING_DIM, input_length=max_length),

#     # GlobalMaxPooling layer to extract relevant features
#     tf.keras.layers.GlobalMaxPool1D(),

#     # First Dense layer with 40 neurons and ReLU activation
#     tf.keras.layers.Dense(64, activation='relu'),

#     # Dropout layer to prevent overfitting
#     tf.keras.layers.Dropout(0.5),

#     # Second Dense layer with 20 neurons and ReLU activation
#     tf.keras.layers.Dense(16, activation='relu'),

#     # Dropout layer to prevent overfitting
#     tf.keras.layers.Dropout(0.4),

#     # Third Dense layer with 10 neurons and ReLU activation
#     tf.keras.layers.Dense(4, activation='relu'),

#     # Dropout layer to prevent overfitting
#     tf.keras.layers.Dropout(0.1),

#     # Final Dense layer with 1 neuron and sigmoid activation for binary classification
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])


model = tf.keras.Sequential([

    tf.keras.layers.InputLayer(input_shape=(max_length,)),

    # Embedding layer for creating word embeddings
    tf.keras.layers.Embedding(vocab_size, TOTAL_EMBEDDING_DIM, input_length=max_length),

    # Conv1D layer for pattern recognition model and extract the feature from the vectors
    tf.keras.layers.Conv1D(filters=128, kernel_size=2),
    
    # GlobalMaxPooling layer to extract relevant features
    tf.keras.layers.GlobalMaxPool1D(),

    # First Dense layer with 64 neurons and ReLU activation
    tf.keras.layers.Dense(16, activation='relu'),

    # Dropout layer to prevent overfitting
    tf.keras.layers.Dropout(0.5),

    # Final Dense layer with 1 neuron and sigmoid activation for binary classification
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# learning_rate = 0.00001, beta_1=0.99, beta_2=0.9999
# compile the model
model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), metrics=["accuracy"])



# summarize the model
print(f"\n{model.summary()}")
print("\n # Wait just Fitting model on training data")
plot_model(model, to_file='summary.png', show_shapes=True, show_layer_names=True, dpi=1000)



# splits into traint & test
tweet_train, tweet_test, labeled_train, labeled_test = train_test_split(padded_docs, cleaned_dataset["sarcasm"].to_numpy(), test_size=0.20, random_state=42)



# sm = SMOTE()
# tweet_train, labeled_train = sm.fit_resample(tweet_train, labeled_train) # type: ignore



# fit the model
result = model.fit(tweet_train, labeled_train, epochs = 40, validation_data = (tweet_test, labeled_test), verbose = 1, callbacks=[callback]) # type: ignore



# get Classification report
print()
predicted = np.round(model.predict(tweet_test))
accuracy = accuracy_score(labeled_test, predicted)
report = classification_report(labeled_test, predicted, target_names=["non-Sarcasm", "Sarcasm"])

print(f"\n{report}\n")



confusionMatrix = confusion_matrix(labeled_test, predicted)

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
val_acc = result.history['val_accuracy']
loss = result.history['loss']
val_loss = result.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'g', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(f"Training vs validation accuracy", dpi=1000)

plt.close()

plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(f"Training vs validation loss", dpi=1000)

plt.close()