import math
import os
import sys

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(r"..\..\sysPath")
os.chdir(dname)

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from arabert import ArabertPreprocessor
import shutil
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import preProcessData #type: ignore
import tensorflow as tf
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_NAME = "aubmindlab/bert-base-arabertv02"

# Load the tokenizer and pre[rpcessor]
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
preprocess = ArabertPreprocessor(model_name=MODEL_NAME)

# Tokenize the dataset
dataset = pd.read_csv(r"../../Datasets/full Dataset.csv")
tweets = dataset["tweet"].apply(lambda x: preprocess.preprocess(x)).tolist()

encoded_tweets = tokenizer(tweets, padding=True, truncation=True, max_length=128, return_tensors="tf")

# Labels
# Assume 0 for 'not sarcastic' and 1 for 'sarcastic'
labels = dataset['sarcasm'].values  # Replace with your labels
labels = tf.keras.utils.to_categorical(labels, num_classes=2)

# Load the AraBERT model with a classification head
model = TFAutoModelForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv02", num_labels=2)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Convert to TensorFlow dataset object
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(input_ids=encoded_tweets['input_ids'], attention_mask=encoded_tweets['attention_mask']),
    labels
)).shuffle(len(tweets)).batch(8)

# Train the model
model.fit(train_dataset, epochs=3)

# Evaluate the model
# ... (your evaluation code here)

# To make predictions
# predictions = model.predict(encoded_tweets['input_ids'], attention_mask=encoded_tweets['attention_mask'])

'''
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer
from arabert import ArabertPreprocessor

# Load the dataset
dataset = pd.read_csv(r"path_to_your/full Dataset.csv")

# Convert boolean 'sarcasm' labels to integers
dataset['sarcasm'] = dataset['sarcasm'].astype(int)

# Convert labels to categorical format
labels = tf.keras.utils.to_categorical(dataset['sarcasm'], num_classes=2)

# Prepare the tokenizer and preprocessor
MODEL_NAME = "aubmindlab/bert-base-arabertv02"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
preprocess = ArabertPreprocessor(model_name=MODEL_NAME)

# Tokenize the tweets
tweets = dataset["tweet"].apply(lambda x: preprocess.preprocess(x)).tolist()
encoded_tweets = tokenizer(tweets, padding=True, truncation=True, max_length=128, return_tensors="tf")

# Ensure that the number of labels matches the number of tweets
assert len(labels) == len(encoded_tweets['input_ids'])

# Create the TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(input_ids=encoded_tweets['input_ids'], attention_mask=encoded_tweets['attention_mask']),
    labels
)).shuffle(len(tweets)).batch(8)

'''