import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_input_sentence(input_sentence, tokenizer, max_length):
    sequence = tokenizer.texts_to_sequences([input_sentence])[0]
    padded_sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
    return padded_sequence

def postprocess_output_sequence(output_sequence, tokenizer):
    sampled_indexes = [np.argmax(token) for token in output_sequence[0]]
    sampled_indexes = [index for index in sampled_indexes if index != 0]
    sampled_tokens = [tokenizer.index_word[index] for index in sampled_indexes]
    return ' '.join(sampled_tokens)

def inference(model, source_tokenizer, target_tokenizer, max_source_length, units):
    input_sentences = ["Toca do Coelho"]

    for input_sentence in input_sentences:
        input_sequence = preprocess_input_sentence(input_sentence, source_tokenizer, max_source_length)

        initial_states = [np.zeros((1, units))]
        output_sequence = model.predict([input_sequence] + initial_states)
        translation = postprocess_output_sequence(output_sequence, target_tokenizer)

        print(f"Input: {input_sentence}\nTranslation: {translation}\n")

if __name__ == "__main__":
    model = tf.keras.models.load_model('PAST-2.keras')

    with open('source_tokenizer.pkl', 'rb') as source_file:
        source_tokenizer = pickle.load(source_file)

    with open('target_tokenizer.pkl', 'rb') as target_file:
        target_tokenizer = pickle.load(target_file)

    units = 512

    max_source_length = 43

    inference(model, source_tokenizer, target_tokenizer, max_source_length, units)