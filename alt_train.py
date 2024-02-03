import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your TSV dataset
df = pd.read_csv("dataset/train.tsv", sep="\t", header=None, names=["Portuguese", "Angrarosskesh"])

# Tokenize the sentences
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(df["Portuguese"].tolist() + df["Angrarosskesh"].tolist())

# Convert text to sequences
input_sequences = tokenizer.texts_to_sequences(df["Portuguese"].tolist())
target_sequences = tokenizer.texts_to_sequences(df["Angrarosskesh"].tolist())

# Pad sequences
max_seq_length = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding="post")
target_sequences = pad_sequences(target_sequences, maxlen=max_seq_length, padding="post")

# Define the model
input_seq = Input(shape=(max_seq_length,))
encoder = LSTM(256, return_state=True)
_, state_h, state_c = encoder(input_seq)
decoder = LSTM(256, return_sequences=True, return_state=True)
decoder_output, _, _ = decoder(target_sequences, initial_state=[state_h, state_c])
output_layer = Dense(len(tokenizer.word_index) + 1, activation="softmax")
output = output_layer(decoder_output)

model = Model(input_seq, output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Train the model
model.fit(input_sequences, target_sequences, epochs=10, batch_size=64)

# Save the trained model
model.save("path/to/save/model")
