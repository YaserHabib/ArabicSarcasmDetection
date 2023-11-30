import os
import sys

from arabert import ArabertPreprocessor
from transformers import AutoTokenizer
import tensorflow as tf
import numpy as np

def tokenize_and_preprocess_single(text, tokenizer, preprocess):
    # Apply the preprocessing to the single text string
    preprocessed_text = preprocess.preprocess(text)

    # Tokenize the preprocessed text
    encoded = tokenizer(preprocessed_text, padding=True, truncation=True, max_length=128, return_tensors="tf")

    return encoded

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv02')
model = tf.keras.models.load_model(r'C:\Users\Perseus\Documents\GitHub\ArabicSarcasmDetection\Deep Learning\araBert\SarcasmAraBERT(20 Epoch)')
preprocess = ArabertPreprocessor(model_name='aubmindlab/bert-base-arabertv02')
# Tokenize your input text
text = '!!أكيد، لأن الانترنت في الصحراء أسرع من المدينة'
notsarcastic = 'أنا كسَرت الأصنامَ'
inputs = tokenize_and_preprocess_single(text, tokenizer, preprocess)

# Convert to TensorFlow Tensors manually
input_ids = tf.convert_to_tensor(inputs['input_ids'], dtype=tf.int32)
attention_mask = tf.convert_to_tensor(inputs['attention_mask'], dtype=tf.int32)
token_type_ids = tf.convert_to_tensor(inputs['token_type_ids'], dtype=tf.int32)

# Prepare the dictionary in the format expected by the model
model_inputs = {
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'token_type_ids': token_type_ids
}

# Load the model and make predictions (assuming 'loaded_model' is your model)
predictions = model(model_inputs)

logits = predictions['logits']

probabilities = tf.nn.softmax(logits, axis=-1)
predicted_class = tf.argmax(probabilities, axis=-1)
predicted_class_numpy = predicted_class.numpy()


print(text)
if predicted_class_numpy == 0:
    print("The text is not sarcastic 🔴")
else:
    print("The text is sarcastic 🟢")

