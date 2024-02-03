import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
data = """É um dia bonito.\tEixh daita krasiyvo.
Das cinzas, pois, foi nascido o humano\tVroet doista, forva, vatexh bruanivet manvek
Poema do principío\tPrinkipaxhpoemet
Poema\tPoemet
Princípio\tPrinkipaxh
Eu fui ao parque\tIg puenxhet ov park
O ser humano é incrível\tDa manvek eixh inveratvy
Ele disse que faria isso\tRivmy parosket ev doatezh vit
Atirei nele, sim\tXhotet ov rivmy, iaxh
Vai chover amanhã\tEixhetav rany tovarov
Alice no País das Maravilhas\tAlice ov Naskan av Vantezhap
Lewis Carroll\tLewis Carroll
Capítulo I Descendo a Toca do Coelho\tXhapit I Dova ov Huta av Rovit
Alice estava começando a ficar muito cansada de sentar-se ao lado de sua irmã no banco e de não ter nada para fazer: uma ou duas vezes havia espiado o livro que a irmã estava lendo, mas não havia imagens nem diálogos nele.\tAlice eixhet vegheva ov veire toirev av xhit ov xhade ov yurv xhista ov xhitprank ant noy hat notanv vo doy: ona ot doa tomap xhetet ixhpatet vat yurv xhista eixhet rida, brat noy eixhet frigavap nivoy tokep ovit.
Não havia nada de tão extraordinário nisso; nem Alice achou assim tão fora do normal ouvir o Coelho\tNoy eixhet notanv av zo ekhxhtarodner ovit; noy Alice vandet zo okt av normal riar da rovit
Eu tive um problema na manhã que se sucedeu ao dia anterior fatídico\tYg hatet ona probezhamit ov varov vat xhakvidet da toirevata dhate vifor
"""

# Split data into source (Portuguese) and target (Fictional language) sentences
pairs = [line.split('\t') for line in data.split('\n') if line]
source_sentences, target_sentences = zip(*pairs)

# Tokenize source and target sentences
source_tokenizer = Tokenizer(filters='')
source_tokenizer.fit_on_texts(source_sentences)
source_vocab_size = len(source_tokenizer.word_index) + 1

target_tokenizer = Tokenizer(filters='')
target_tokenizer.fit_on_texts(target_sentences)
target_vocab_size = len(target_tokenizer.word_index) + 1

with open('source_tokenizer.pkl', 'wb') as source_file:
    pickle.dump(source_tokenizer, source_file)

with open('target_tokenizer.pkl', 'wb') as target_file:
    pickle.dump(target_tokenizer, target_file)

# Convert sentences to sequences
source_sequences = source_tokenizer.texts_to_sequences(source_sentences)
target_sequences = target_tokenizer.texts_to_sequences(target_sentences)

# Pad sequences
source_padded = pad_sequences(source_sequences)
max_source_length = source_padded.shape[1]
print(max_source_length)
target_padded = pad_sequences(target_sequences, padding='post')

def train():
    # Define the model
    embedding_dim = 256
    units = 512

    # Encoder
    encoder_inputs = tf.keras.layers.Input(shape=(None,))
    encoder_embedding = tf.keras.layers.Embedding(source_vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm = tf.keras.layers.LSTM(units, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = tf.keras.layers.Input(shape=(None,))
    decoder_embedding = tf.keras.layers.Embedding(target_vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    # Output layer
    decoder_dense = tf.keras.layers.Dense(target_vocab_size, activation='softmax')
    outputs = decoder_dense(decoder_outputs)

    # Model
    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit([source_padded, target_padded[:, :-1]], target_padded[:, 1:], epochs=11, batch_size=64, validation_split=0.2)

    model.evaluate([source_padded, target_padded[:, :-1]], target_padded[:, 1:])

    # Save the model
    model.save('translator_model.keras')

train()

