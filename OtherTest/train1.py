import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, Dropout
from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import glob

# Custom tokenizer function to handle RPG III code
def tokenize_rpgiii_code(code_line):
    # Determine the starting position based on the 'type' field length
    type_field = code_line[5:7].strip()
    if len(type_field) == 1:
        start_pos = 6
    else:
        start_pos = 7

    # Define tokens based on the starting position
    tokens = {
        'type': type_field,
        'control': code_line[start_pos:start_pos+2].strip(),
        'indicator1': code_line[start_pos+2:start_pos+5].strip(),
        'indicator2': code_line[start_pos+5:start_pos+8].strip(),
        'indicator3': code_line[start_pos+8:start_pos+11].strip(),
        'factor1': code_line[start_pos+11:start_pos+21].strip(),
        'opcode': code_line[start_pos+21:start_pos+26].strip(),
        'factor2': code_line[start_pos+26:start_pos+36].strip(),
        'result': code_line[start_pos+36:start_pos+42].strip(),
        'len': code_line[start_pos+42:start_pos+45].strip(),
        'de': code_line[start_pos+45:start_pos+47].strip(),
        'hi': code_line[start_pos+47:start_pos+49].strip(),
        'lo': code_line[start_pos+49:start_pos+51].strip(),
        'eq': code_line[start_pos+51:start_pos+53].strip(),
        'comments': code_line[start_pos+53:].strip()
    }

    return tokens

# Transformer Encoder Layer
def transformer_encoder(vocab_size, embed_dim, num_heads, ff_dim, max_sequence_len, rate=0.1):
    inputs = Input(shape=(max_sequence_len,))
    embedding_layer = Embedding(vocab_size, embed_dim)(inputs)
    transformer_block = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(embedding_layer, embedding_layer)
    transformer_block = Dropout(rate)(transformer_block)
    transformer_block = LayerNormalization(epsilon=1e-6)(transformer_block + embedding_layer)
    # Removed GlobalAveragePooling1D, now transformer_block is 3D tensor
    return Model(inputs, transformer_block)

# Transformer Decoder Layer
def transformer_decoder(vocab_size, embed_dim, num_heads, ff_dim, max_sequence_len, rate=0.1):
    inputs = Input(shape=(max_sequence_len, embed_dim))  # This shape should match the encoder output
    transformer_block = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    transformer_block = Dropout(rate)(transformer_block)
    transformer_block = LayerNormalization(epsilon=1e-6)(transformer_block + inputs)
    # The output layer should map the transformer block output to the vocabulary size
    outputs = Dense(vocab_size, activation="softmax")(transformer_block)
    return Model(inputs, outputs)

# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
else:
    print("No GPU available, training on CPU.")

# Assuming all your files are in the same directory and have the '.txt' extension
file_directory = '/content/sample'
file_pattern = os.path.join(file_directory, '*.txt')

# List of DataFrames, one for each file
dataframes = []

# Loop through all the text files in the directory
for filepath in glob.glob(file_pattern):
    # Load the dataset from the current file
    df = pd.read_csv(filepath, names=['sentence'], sep='\t', usecols=[0])
    # Tokenize each line of RPG III code
    df['tokens'] = df['sentence'].apply(lambda x: list(tokenize_rpgiii_code(x).values()))
    # Append the DataFrame to the list
    dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Load the dataset and tokenize each line of RPG III code
df['tokens'] = df['sentence'].apply(lambda x: list(tokenize_rpgiii_code(x).values()))

# Flatten the list of tokenized sentences to fit the tokenizer
all_tokens = [token for sublist in combined_df['tokens'].tolist() for token in sublist]

# Tokenize the text using the custom token list
tokenizer = Tokenizer(filters='', oov_token='<OOV>')  # Disable default filters
tokenizer.fit_on_texts(all_tokens)
sequences = tokenizer.texts_to_sequences(combined_df['tokens'].tolist())
max_sequence_len = max([len(seq) for seq in sequences])
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# Pad sequences
data = pad_sequences(sequences, maxlen=max_sequence_len, padding='post')

# Define hyperparameters
embed_dim = 64  # Embedding size for each token
num_heads = 5  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer

# Define the encoder model
encoder = transformer_encoder(vocab_size, embed_dim, num_heads, ff_dim, max_sequence_len)

# Define the decoder model
decoder = transformer_decoder(vocab_size, embed_dim, num_heads, ff_dim, max_sequence_len)

# Build the autoencoder model
autoencoder_input = Input(shape=(max_sequence_len,))
encoded_seq = encoder(autoencoder_input)

# Ensure that the encoder outputs the correct shape (batch_size, sequence_length, embed_dim)
# If the encoder does not output this shape, you need to modify the encoder architecture accordingly

# Pass the encoded sequence directly to the decoder without reshaping
decoded_seq = decoder(encoded_seq)

autoencoder = Model(autoencoder_input, decoded_seq)
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')

# Prepare target data for the autoencoder
target_data = tf.keras.utils.to_categorical(data, vocab_size)

# Train the autoencoder
history = autoencoder.fit(data, target_data,
                          epochs=32,  # Correct number of epochs
                          batch_size=8)

# Plot the loss over epochs
plt.plot(history.history['loss'], label='Training loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Example RPG III code line
rpgiii_line = "     C  N50 91             MOVEL'*INCONNU'EL,I"
              
# Tokenize the RPG III code line
rpgiii_tokens = list(tokenize_rpgiii_code(rpgiii_line).values())

# Convert the tokenized RPG III code to a sequence using the tokenizer
sequence = tokenizer.texts_to_sequences([rpgiii_tokens])

# Pad the sequence to the maximum sequence length
padded_sequence = pad_sequences(sequence, maxlen=max_sequence_len, padding='post')

# The padded_sequence can now be used as input to the autoencoder
autoencoder_input = padded_sequence[0]

# Now you can use autoencoder_input with the encoder
encoded_sentence = encoder.predict(np.array([autoencoder_input]))  # Wrap in np.array

# Decode the sentence
decoded_sentence = decoder.predict(encoded_sentence)

# Convert the sequence of tokens to words
decoded_words = []
for i in decoded_sentence[0]:
    token_index = np.argmax(i)
    for word, index in word_index.items():
        if index == token_index:
            decoded_words.append(word)
            break

# Join the words to form the decoded sentence
decoded_sentence = ' '.join(decoded_words)

# Evaluate the model on the test data
test_results = autoencoder.evaluate(test_data, test_labels)

# Save the test results to a CSV file
test_results_df = pd.DataFrame(test_results, index=['loss', 'accuracy'])  # Adjust index based on what you're evaluating
test_results_df.to_csv('test_results.csv')

# Save the entire model to a file
autoencoder.save('autoencoder_model.h5')  # Saves the model in HDF5 format

# Alternatively, save only the model weights
autoencoder.save_weights('autoencoder_weights.h5')

# If you want to save in TensorFlow SavedModel format (recommended for TensorFlow 2.x)
autoencoder.save('autoencoder_saved_model')  # This will create a directory with the SavedModel

# To load the model back, you can use
loaded_model = tf.keras.models.load_model('autoencoder_model.h5')  # or 'autoencoder_saved_model' for SavedModel

print('Decoded sentence:', decoded_sentence)