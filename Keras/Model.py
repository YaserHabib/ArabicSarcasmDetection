import os
import re
import nltk
import shutil
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pyarabic.normalize as Normalize

from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from keras.callbacks import TensorBoard
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model, pad_sequences

from sklearn.svm import SVC
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from numpy import array
from matplotlib import style
from gensim.models import KeyedVectors
from transformers import MarianMTModel, MarianTokenizer

le = LabelEncoder()
style.use("ggplot")



# nltk.download('punkt')  # download punkt tokenizer if not already downloaded
# nltk.download('stopwords') # download stopwords if not already downloaded
# nltk.download('averaged_perceptron_tagger')
warnings.filterwarnings(action = 'ignore')



path = r"C:\Users\Mohamed\Documents\Fall 2023 - 2024\Senior Project in CS\Keras"
if os.path.isdir("logs"):
    shutil.rmtree("logs")
callback = TensorBoard(log_dir=rf'{path}\logs/', histogram_freq=1)



def remove_emojis(text):
    emoji_pattern = re.compile("["
                                    u"\U0001F600-\U0001F64F"  # emoticons
                                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                    u"\U00002702-\U000027B0"
                                    u"\U000024C2-\U0001F251"
                                    u"\U0001F90C-\U0001F93A"  # Supplemental Symbols
                                    u"\U0001F93C-\U0001F945"  # and
                                    u"\U0001F947-\U0001F9FF"  # Pictographs
                                "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)



def removeConsecutiveDuplicates(text):
    # Replace any group of two or more consecutive characters with just one
    #clean = re.sub(r'(\S)(\1+)', r'\1', text, flags=re.UNICODE)

    clean = re.sub(r'(\S)(\1{2,})', r'\1', text, flags=re.UNICODE)
    #This one only replaces it if there are more than two duplicates. For example, الله has 2 لs but we don't want it removed

    return clean



def removeEnglish(text):
    return re.sub(r"[A-Za-z0-9]+","",text)



def lemmatizeArabic(text):
    """
    This function takes an Arabic word as input and returns its lemma using NLTK's ISRI stemmer
    """
    # Create an instance of the ISRI stemmer
    stemmer = ISRIStemmer()
    # Apply the stemmer to the word
    lemma = stemmer.stem(text)
    return lemma



def removeStopwords(text):
    # Tokenize the text into wordsz
    words = nltk.word_tokenize(text)
    # Get the Arabic stop words from NLTK
    stop_words = set(stopwords.words('arabic'))
    # Remove the stop words from the list of words
    words_filtered = [word for word in words if word.lower() not in stop_words]
    # Join the words back into a string
    clean = ' '.join(words_filtered)
    return clean



def removePunctuation(text):
    # Define the Arabic punctuation regex pattern
    arabicPunctPattern = r'[؀-؃؆-؊،؍؛؞]'
    engPunctPattern = r'[.,;''`~:"]'
    # Use re.sub to replace all occurrences of Arabic punctuation with an empty string
    clean = re.sub(arabicPunctPattern + '|' + engPunctPattern, '', text)
    return clean



def tokenizeArabic(text):
    # Tokenize the text using the word_tokenize method and return the list of tokenized words
    return word_tokenize(text)



def cleanData(dataset):
    dataset = dataset.drop_duplicates(subset=["tweet"])
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)

    for index, tweet in enumerate(dataset["tweet"].tolist()):
        #standard tweet cleaning
        clean = re.sub(r"(http[s]?\://\S+)|([\[\(].*[\)\]])|([#@]\S+)|\n", "", tweet)
        
        #Test to see if they're useful or not
        clean = remove_emojis(clean)
        clean = removeConsecutiveDuplicates(clean)

        # mandatory arabic preprocessing
        clean = Normalize.normalize_searchtext(clean)
        clean = removeEnglish(clean)
        clean = lemmatizeArabic(clean)
        clean = removeStopwords(clean)
        clean = removePunctuation(clean)

        # clean = tokenizeArabic(clean)
        dataset.loc[index, "tweet"] = clean # replace the old values with the cleaned one.

    return dataset



def tokenization(dataset):
    for index, tweet in enumerate(dataset[["tweet"]].values.tolist()):
        tokenizedTweet = tokenizeArabic(*tweet)
        dataset.at[index, "tweet"] = tokenizedTweet

    dataset["sentiment"] = le.fit_transform(dataset["sentiment"])
    dataset["dialect"] = le.fit_transform(dataset["dialect"])
    dataset["sarcasm"] = le.fit_transform(dataset["sarcasm"])

    return dataset



def preProcessData(dataset):

    data = cleanData(dataset.copy(deep=True))
    print("\n----------         cleanData Done!         ----------\n")

    # dataAugmentation()
    print("\n---------- dataAugmentation done in a separate file ----------\n")

    data = tokenization(data.copy(deep=True))
    print("\n----------     dataTokenization Done!      ----------\n")

    return data



dataset = pd.read_csv(r"C:\Users\Mohamed\Documents\Fall 2023 - 2024\Senior Project in CS\afterBackTrans.csv")
dataset.info()
# display the dataset before the pre-processing
print(f"\n{dataset.head()}")



# cleaned_dataset.head()
cleaned_dataset = preProcessData(dataset.copy(deep=True))
# cleaned_dataset.head()
cleaned_dataset.to_excel('cleanedset.xlsx', index=False)
# display the dataset after pre-processing
print(f"{cleaned_dataset.head()}\n")



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
# model = Sequential()
# embedding_layer = tf.keras.layers.Embedding(vocab_size, TOTAL_EMBEDDING_DIM, weights=[embedding_matrix], input_length=4, trainable=False)
# input_placeholder= tf.keras.Input(shape=(max_length,), dtype = "int32")
# input_embedding = embedding_layer(input_placeholder)
# lstm = tf.keras.layers.LSTM(units = 10, activation = "relu")(input_embedding)
# preds = tf.keras.layers.Dense(1, activation = "sigmoid", name = "activation")(lstm)
# model = tf.keras.models.Model(inputs = input_placeholder, outputs = preds)
# # model.add(tf.keras.layers.Dense(10, input_dim=300, activation='relu'))
# # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, TOTAL_EMBEDDING_DIM, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])



# compile the model
model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999), metrics=["accuracy"])



# summarize the model
print(f"\n{model.summary()}")
print("\n # Wait just Fitting model on training data")
plot_model(model, to_file='summary.png', show_shapes=True, show_layer_names=True)



labels = cleaned_dataset[["sentiment"]].copy()



# splits into traint & test
tweet_train, tweet_test, labeled_train, labeled_test = train_test_split(padded_docs, labels.to_numpy(), test_size=0.35, shuffle=True)



# fit the model
model.fit(tweet_train, labeled_train, epochs = 5, verbose = 1, callbacks=[callback]) # type: ignore



#evaluate the model
print(model.predict(tweet_test))

trainLoss, trainScore = model.evaluate(tweet_train, labeled_train)
testLoss, testScore = model.evaluate(tweet_test, labeled_test)
labelPredicted = np.round(model.predict(tweet_test))

print()

print(f"keras score on the training dataset: {trainScore:.2f}")
print(f"keras score on the test dataset:     {testScore:.2f}\n")

print(classification_report(labeled_test, labelPredicted, target_names=["Class: 0", "Class: 1", "Class: 2"]))



confusionMatrix = confusion_matrix(labeled_test, labelPredicted)

confMatrix_display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=[0, 1, 2])
confMatrix_display.plot()
confMatrix_display.figure_.savefig(fname = rf"{path}\sarcasmModel.png", dpi=1000)

plt.show()