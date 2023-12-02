import math
import os
import sys

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(r"..\..\sysPath")
os.chdir(dname)

results_dir = os.path.join(os.path.dirname(__file__), "Arabert Results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AdamW
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
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, LearningRateScheduler
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Redefine the tokenize_and_preprocess function
def tokenize_and_preprocess(dataframe, tokenizer, preprocess):
    tweets = dataframe["tweet"].apply(lambda x: preprocess.preprocess(x)).tolist()
    encoded = tokenizer(tweets, padding=True, truncation=True, max_length=128, return_tensors="tf")
    labels = tf.keras.utils.to_categorical(dataframe['sarcasm'].values, num_classes=2)
    return encoded, labels

MODEL_NAME = "aubmindlab/bert-base-arabertv2"

# Load the tokenizer and preprocessor]
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
preprocess = ArabertPreprocessor(model_name=MODEL_NAME)

SPLIT_VALUE = 0.20  
# Tokenize the dataset
dataset = pd.read_csv(r"../../Datasets/full Dataset.csv")
train_df, test_df = train_test_split(dataset, test_size=SPLIT_VALUE, random_state=42)

# Split the training data further to create a validation set
train_df, val_df = train_test_split(train_df, test_size=SPLIT_VALUE, random_state=42)

# Tokenize and preprocess training and testing sets
train_encoded, train_labels = tokenize_and_preprocess(train_df, tokenizer, preprocess)
val_encoded, val_labels = tokenize_and_preprocess(val_df, tokenizer, preprocess)
test_encoded, test_labels = tokenize_and_preprocess(test_df, tokenizer, preprocess)

BATCH_SIZE = 8

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': train_encoded['input_ids'], 'attention_mask': train_encoded['attention_mask']},
    train_labels
)).shuffle(len(train_df)).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': val_encoded['input_ids'], 'attention_mask': val_encoded['attention_mask']},
    val_labels
)).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': test_encoded['input_ids'], 'attention_mask': test_encoded['attention_mask']},
    test_labels
)).batch(BATCH_SIZE)


# Load the AraBERT model with a classification head
model = TFAutoModelForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv02", num_labels=2)
'''config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2, hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5)
model = TFAutoModelForSequenceClassification.from_config(config)'''

model.summary()
num_layers = len(model.layers)
print(f"Total number of layers in the BERT Layers:",len(model.layers[0].encoder.layer))
'''
#Freeze AraBERT Layerst to prevent overfitting(Keep Top Layers)
nonFrozenLayers = 5
totalLayers = len(model.layers[0].encoder.layer)  # Total number of layers in the encoder

for layer in model.layers[0].encoder.layer[:totalLayers-nonFrozenLayers]:
    layer.trainable = False
'''
#Keep Bottom Layers
nonFrozenLayers = 4

for layer in model.layers[0].encoder.layer[:-nonFrozenLayers]:
    layer.trainable = False
model.summary()

#Hyperparameters
EPOCH = 20
LEARNING_RATE = 5e-7
WEIGHT_DECAY = 0.000

# Compile the model
#optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam', decay=WEIGHT_DECAY)
#optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, name='Adam', decay=WEIGHT_DECAY)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, name='Adam')
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    mode='min',           
    restore_best_weights=True
)

def step_decay(epoch):
    initial_lrate = 5e-5
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

# Train the model with validation data
#history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCH, callbacks = [early_stopping])
history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCH, callbacks = [early_stopping])

#model.save('./Sarcasm_araBERT_NEW')

with open('arabert_model.pkl', 'wb') as file:
        pickle.dump(model, file)

print("Model have been saved as pickle file.")

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)

predictions = model.predict(test_dataset)
predicted_classes = np.argmax(predictions.logits, axis=1)
true_classes = np.argmax(test_labels, axis=1)
print(classification_report(true_classes, predicted_classes))

# Accuracy Plot
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
filename_acc = f"accuracy_epoch {EPOCH}_lr {LEARNING_RATE}_wd {WEIGHT_DECAY}.png"
full_path_acc = os.path.join(results_dir, filename_acc)
plt.savefig(full_path_acc, dpi=300)
plt.close()

# Loss Plot
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
filename_loss = f"loss_epoch {EPOCH}_lr {LEARNING_RATE}_wd {WEIGHT_DECAY}.png"
full_path_loss = os.path.join(results_dir, filename_loss)
plt.savefig(full_path_loss, dpi=300)  # Saves the loss plot
plt.close()

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
filename_cm = f"confusion_matrix_epoch {EPOCH}_lr {LEARNING_RATE}_wd {WEIGHT_DECAY}.png"
full_path_cm = os.path.join(results_dir, filename_cm)
plt.savefig(full_path_cm, dpi=300)
plt.close()

