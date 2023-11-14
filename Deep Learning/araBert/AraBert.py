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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_NAME = "aubmindlab/bert-base-arabertv02"

# Load the tokenizer and pre[rpcessor]
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
preprocess = ArabertPreprocessor(model_name=MODEL_NAME)

# Tokenize the dataset
dataset = pd.read_csv(r"../../Datasets/full Dataset.csv")
train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)

# Function to tokenize and preprocess a dataset
def tokenize_and_preprocess(dataframe):
    tweets = dataframe["tweet"].apply(lambda x: preprocess.preprocess(x)).tolist()
    encoded = tokenizer(tweets, padding=True, truncation=True, max_length=128, return_tensors="tf")
    labels = tf.keras.utils.to_categorical(dataframe['sarcasm'].values, num_classes=2)
    return encoded, labels

# Tokenize and preprocess training and testing sets
train_encoded, train_labels = tokenize_and_preprocess(train_df)
test_encoded, test_labels = tokenize_and_preprocess(test_df)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': train_encoded['input_ids'], 'attention_mask': train_encoded['attention_mask']},
    train_labels
)).shuffle(len(train_df)).batch(8)

test_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': test_encoded['input_ids'], 'attention_mask': test_encoded['attention_mask']},
    test_labels
)).batch(8)


# Load the AraBERT model with a classification head
model = TFAutoModelForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv02", num_labels=2)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=3)

# Evaluate the model
# ... (your evaluation code here)

# To make predictions
# predictions = model.predict(encoded_tweets['input_ids'], attention_mask=encoded_tweets['attention_mask'])
