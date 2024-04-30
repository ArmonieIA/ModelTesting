import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, Dropout
from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from Parser.tokenizer import tokenize_rpgiii_code
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import glob
import concurrent.futures
import gc

# Transformer Encoder Layer
def transformer_encoder(vocab_size, embed_dim, num_heads, ff_dim, max_sequence_len, rate=0.1):
    inputs = Input(shape=(max_sequence_len,))
    embedding_layer = Embedding(vocab_size, embed_dim)(inputs)
    transformer_block = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(embedding_layer, embedding_layer)
    transformer_block = Dropout(rate)(transformer_block)
    # Cast the embedding_layer to the same type as transformer_block before addition
    transformer_block = LayerNormalization(epsilon=1e-6)(transformer_block + tf.cast(embedding_layer, transformer_block.dtype))
    return Model(inputs, transformer_block)

# Transformer Decoder Layer
def transformer_decoder(vocab_size, embed_dim, num_heads, ff_dim, max_sequence_len, rate=0.1):
    inputs = Input(shape=(max_sequence_len, embed_dim))  # This shape should match the encoder output
    transformer_block = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    transformer_block = Dropout(rate)(transformer_block)
    # Cast the inputs to the same type as transformer_block before addition
    transformer_block = LayerNormalization(epsilon=1e-6)(transformer_block + tf.cast(inputs, transformer_block.dtype))
    outputs = Dense(vocab_size, activation="softmax")(transformer_block)
    return Model(inputs, outputs)

# Function to process a single file
def process_file(filepath):
    df = pd.read_csv(filepath, names=['sentence'], sep='\t', usecols=[0])
    # Load the dataset and tokenize each line of RPG III code
    df['tokens'] = df['sentence'].apply(lambda x: list(tokenize_rpgiii_code(x).values()))
    return df

# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
else:
    print("No GPU available, training on CPU.")

# Assuming all your files are in the same directory and have the '.txt' extension
file_directory = '/content/sample'
file_pattern = os.path.join(file_directory, '*.txt')

# List to hold the results
dataframes = []

# Use ThreadPoolExecutor for parallel processing
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Map the process_file function to all files
    futures = {executor.submit(process_file, filepath): filepath for filepath in glob.glob(file_pattern)}
    for future in concurrent.futures.as_completed(futures):
        # Append the result to the dataframes list
        dataframes.append(future.result())

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

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
embed_dim = 128  # Embedding size for each token
num_heads = 5  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer
batch_size = 20 
epoch = 32

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
# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',
                    metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.metrics.Precision(), tf.metrics.Recall()])

# Prepare target data for the autoencoder
target_data = tf.keras.utils.to_categorical(data, vocab_size)

# Use float32 instead of float64 if possible
# Ensure your data is in float32 when creating numpy arrays, etc.

# Enable mixed precision training
mixed_precision = tf.keras.mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Assuming 'data' is your input features and 'target_data' is your labels
train_data, test_data, train_labels, test_labels = train_test_split(
    data, target_data, test_size=0.2, random_state=42
)


# Train the autoencoder with a smaller batch size and mixed precision
history = autoencoder.fit(train_data, train_labels,
                          epochs=epoch,
                          batch_size=batch_size)

# Plot the loss over epochs
plt.plot(history.history['loss'], label='Training loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Example RPG III code line
rpgiii_line = "     C                     Z-ADD*ZEROS    NOAFF"
              
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

test_results_2d = [test_results]  # This makes it a list of lists

# Now create the DataFrame with the correct shape
test_results_df = pd.DataFrame([test_results], columns=['loss', 'accuracy', 'precision', 'recall'])

# Add the RPG III code line and the decoded sentence to the test results DataFrame
test_results_df['rpgiii_line'] = rpgiii_line
test_results_df['decoded_sentence'] = decoded_sentence

# Define the CSV file path
csv_file_path = 'test_results.csv'

# Check if the CSV file already exists
if os.path.isfile(csv_file_path):
    # If it exists, load the existing data and concatenate the new test results
    existing_df = pd.read_csv(csv_file_path)
    updated_df = pd.concat([existing_df, test_results_df], ignore_index=True)
else:
    # If it does not exist, just use the new test results DataFrame
    updated_df = test_results_df

# Save the updated DataFrame to the CSV file
updated_df.to_csv(csv_file_path, index=False)

# Save the entire model to a file
autoencoder.save('autoencoder_model.h5')  # Saves the model in HDF5 format

# Alternatively, save only the model weights
autoencoder.save_weights('autoencoder_weights.h5')

# If you want to save in TensorFlow SavedModel format (recommended for TensorFlow 2.x)
autoencoder.save('autoencoder_saved_model')  # This will create a directory with the SavedModel

# To load the model back, you can use
loaded_model = tf.keras.models.load_model('autoencoder_model.h5')  # or 'autoencoder_saved_model' for SavedModel

print('Decoded sentence:', decoded_sentence)

# Clear the TensorFlow session
K.clear_session()

# Delete large variables if they are no longer needed
del autoencoder, encoder, decoder, train_data, test_data, train_labels, test_labels
del combined_df, dataframes, tokenizer, sequences, data, target_data

# Run garbage collection
gc.collect()

# Clear the matplotlib plt to free up memory
plt.clf()
plt.close('all')
