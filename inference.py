import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to preprocess input sentence
def preprocess_input_sentence(input_sentence, tokenizer, max_length):
    sequence = tokenizer.texts_to_sequences([input_sentence])[0]
    padded_sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
    return padded_sequence

# Function to post-process output sequence
def postprocess_output_sequence(output_sequence, tokenizer):
    sampled_indexes = [np.argmax(token) for token in output_sequence[0]]
    # Exclude index 0
    sampled_indexes = [index for index in sampled_indexes if index != 0]
    # Retrieve words from the tokenizer
    sampled_tokens = [tokenizer.index_word[index] for index in sampled_indexes]
    return ' '.join(sampled_tokens)

def inference(model, source_tokenizer, target_tokenizer, max_source_length, units):
    # Sample Portuguese sentences for inference
    input_sentences = ["Toca do Coelho"]

    for input_sentence in input_sentences:
        # Preprocess input sentence
        input_sequence = preprocess_input_sentence(input_sentence, source_tokenizer, max_source_length)

        # Initialize initial states
        initial_states = [np.zeros((1, units))]

        # Perform inference
        output_sequence = model.predict([input_sequence] + initial_states)

        # Post-process output sequence
        translation = postprocess_output_sequence(output_sequence, target_tokenizer)

        print(f"Input: {input_sentence}\nTranslation: {translation}\n")

if __name__ == "__main__":
    # Load the saved model
    model = tf.keras.models.load_model('translator_model.keras')

    # Load source and target tokenizers
    with open('source_tokenizer.pkl', 'rb') as source_file:
        source_tokenizer = pickle.load(source_file)

    with open('target_tokenizer.pkl', 'rb') as target_file:
        target_tokenizer = pickle.load(target_file)

    # Define units
    units = 512

    # Set the maximum source sequence length
    max_source_length = 43  # Update this based on your training data

    # Call the inference function
    inference(model, source_tokenizer, target_tokenizer, max_source_length, units)