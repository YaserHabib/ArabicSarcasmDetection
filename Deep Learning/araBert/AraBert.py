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

# Redefine the tokenize_and_preprocess function
def tokenize_and_preprocess(dataframe, tokenizer, preprocess):
    tweets = dataframe["tweet"].apply(lambda x: preprocess.preprocess(x)).tolist()
    encoded = tokenizer(tweets, padding=True, truncation=True, max_length=128, return_tensors="tf")
    labels = tf.keras.utils.to_categorical(dataframe['sarcasm'].values, num_classes=2)
    return encoded, labels

MODEL_NAME = "aubmindlab/bert-base-arabertv02"

# Load the tokenizer and pre[rpcessor]
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
preprocess = ArabertPreprocessor(model_name=MODEL_NAME)

# Tokenize the dataset
dataset = pd.read_csv(r"../../Datasets/full Dataset.csv")
train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)

# Split the training data further to create a validation set
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Tokenize and preprocess training and testing sets
train_encoded, train_labels = tokenize_and_preprocess(train_df, tokenizer, preprocess)
val_encoded, val_labels = tokenize_and_preprocess(val_df, tokenizer, preprocess)
test_encoded, test_labels = tokenize_and_preprocess(test_df, tokenizer, preprocess)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': train_encoded['input_ids'], 'attention_mask': train_encoded['attention_mask']},
    train_labels
)).shuffle(len(train_df)).batch(8)

val_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': val_encoded['input_ids'], 'attention_mask': val_encoded['attention_mask']},
    val_labels
)).batch(8)

test_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': test_encoded['input_ids'], 'attention_mask': test_encoded['attention_mask']},
    test_labels
)).batch(8)


# Load the AraBERT model with a classification head
model = TFAutoModelForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv02", num_labels=2)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-7)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model with validation data
history = model.fit(train_dataset, validation_data=val_dataset, epochs=2)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)

predictions = model.predict(test_dataset)
predicted_classes = np.argmax(predictions.logits, axis=1)
true_classes = np.argmax(test_labels, axis=1)
print(classification_report(true_classes, predicted_classes))

#Accuracy Plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
