import os
import sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(r"C:\Users\Mohamed\Documents\Fall 2023 - 2024\Senior Project\sysPath")
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

from keras.utils import plot_model
from keras.utils import pad_sequences
from gensim.models import KeyedVectors
from sklearn.utils import class_weight
from keras.callbacks import TensorBoard
from imblearn.over_sampling import SMOTE
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings(action = 'ignore')
callback = TensorBoard(log_dir='logs/', histogram_freq=1)

if os.path.isdir("logs"):
    shutil.rmtree("logs")



# dataset, datasetName = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/training_data.csv"), "Original Dataset"
# dataset, datasetName = pd.read_csv(r"../../../Datasets/GPT Dataset.csv"), "GPT Combined Dataset"
dataset, datasetName = pd.read_csv(r"../../../Datasets/full Dataset.csv"), "Full Combined Dataset"
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
# print("encoded_docs:\n",encoded_docs)



# pad documents to a max length of 4 words
max_length = len(max(np.array(dataset["tweet"]), key=len))
padded_docs = pad_sequences(encoded_docs, maxlen = max_length, padding = "post")
print("\npadded_docs:\n",padded_docs)



# load the whole embedding into memory
w2v_embeddings_index = {}
TOTAL_EMBEDDING_DIM = 300
embeddings_file = r"../../../full_grams_cbow_300_twitter/full_grams_cbow_300_twitter.mdl"
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



model = tf.keras.Sequential([

    tf.keras.layers.InputLayer(input_shape=(max_length,)),

    # Embedding layer for creating word embeddings
    tf.keras.layers.Embedding(vocab_size, TOTAL_EMBEDDING_DIM, input_length=max_length, trainable=False),

    # Conv1D layer for pattern recognition model and extract the feature from the vectors
    tf.keras.layers.Conv1D(filters=64, kernel_size=3),
    
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



# splits into traint, validation, and test
train_tweet, test_tweet, train_labels, test_labels = train_test_split(padded_docs, cleaned_dataset["sarcasm"].to_numpy(), test_size=0.20)
train_tweet, val_tweet, train_labels, val_labels = train_test_split(train_tweet, train_labels, test_size=0.20)



# sm = SMOTE()
# tweet_train, labeled_train = sm.fit_resample(tweet_train, labeled_train) # type: ignore



# fit the model
class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weights))
result = model.fit(train_tweet, train_labels, epochs = 20, verbose = 1, validation_data=(val_tweet, val_labels), callbacks=[callback]) # type: ignore



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