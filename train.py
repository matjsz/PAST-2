import pandas as pd
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras.models import Model
from transformers import AutoTokenizer
import tensorflow as tf

# Load your TSV dataset
df = pd.read_csv("dataset/train.tsv", sep="\t", header=None, names=["Portuguese", "Angrarosskesh"])

# Split into train and validation sets
train_size = int(0.8 * len(df))
train_df, val_df = df[:train_size], df[train_size:]

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Define input and output sequences
input_seq = Input(shape=(None,))
target_seq = Input(shape=(None,))

# Embedding layer (you can replace this with pre-trained embeddings)
embedding_dim = 128
vocab_size = 10000  # Replace with your actual vocabulary size
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)

# Encode input sequence
encoder = Bidirectional(LSTM(256, return_state=True))
encoder_output, forward_h, forward_c, backward_h, backward_c = encoder(embedding_layer(input_seq))
state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])

# Decode sequence
decoder = LSTM(256, return_sequences=True, return_state=True)
decoder_output, _, _ = decoder(embedding_layer(target_seq), initial_state=[state_h, state_c])

# Output layer
output_layer = Dense(vocab_size, activation="softmax")
output = output_layer(decoder_output)

# Create the model
model = Model([input_seq, target_seq], output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Tokenize and convert to input IDs
train_input = tokenizer(train_df["Portuguese"].tolist(), padding=True, truncation=True, return_tensors="tf")
train_target = tokenizer(train_df["Angrarosskesh"].tolist(), padding=True, truncation=True, return_tensors="tf")

# # Train the model
model.fit([train_input["input_ids"], train_target["input_ids"]], train_target["input_ids"], epochs=10, batch_size=64)
