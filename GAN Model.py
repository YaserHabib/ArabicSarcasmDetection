import numpy as np
import tensorflow as tf
from keras.layers import Input, Embedding, Conv1D, Flatten, Dense, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Define parameters
seq_length = 50
vocab_size = 10000
embedding_dim = 128
noise_dim = 100
filters = 128

# Generator
def build_generator(seq_length, vocab_size, embedding_dim, noise_dim):
    input_noise = Input(shape=(noise_dim,))
    x = Dense(seq_length * embedding_dim)(input_noise)
    x = Reshape((seq_length, embedding_dim))(x)
    x = Conv1D(filters, kernel_size=5, padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(vocab_size, activation='softmax')(x)
    
    generator = Model(input_noise, x)
    return generator

# Discriminator
def build_discriminator(seq_length, vocab_size, filters):
    input_text = Input(shape=(seq_length,))
    x = Embedding(vocab_size, embedding_dim)(input_text)
    x = Conv1D(filters, kernel_size=5, padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    
    discriminator = Model(input_text, x)
    return discriminator

# Build and compile the discriminator
discriminator = build_discriminator(seq_length, vocab_size, filters)
discriminator.compile(loss=BinaryCrossentropy(), optimizer=Adam(learning_rate=0.0002))

# Build the generator
generator = build_generator(seq_length, vocab_size, embedding_dim, noise_dim)

# Create GAN model
discriminator.trainable = False
gan_input = Input(shape=(noise_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
gan.compile(loss=BinaryCrossentropy(), optimizer=Adam(learning_rate=0.0002))

# Training loop (you need to have a dataset to train on)
# Here, we assume you have a list of tokenized sequences stored in 'text_data'

num_epochs = 10000
batch_size = 64

for epoch in range(num_epochs):
    # Train discriminator
    real_data = np.array(random.sample(text_data, batch_size))
    real_labels = np.ones((batch_size, 1))
    fake_data = generator.predict(np.random.randn(batch_size, noise_dim))
    fake_labels = np.zeros((batch_size, 1))
    
    d_loss_real = discriminator.train_on_batch(real_data, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)
    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    
    # Train generator
    noise = np.random.randn(batch_size, noise_dim)
    g_loss = gan.train_on_batch(noise, real_labels)
    
    print(f"Epoch {epoch}/{num_epochs}, D Loss: {d_loss}, G Loss: {g_loss}")