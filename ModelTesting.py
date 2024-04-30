import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, Dropout
from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import glob
import concurrent.futures
import gc

# Custom tokenizer function to handle RPG III code
def tokenize_rpgiii_code(code_line):
    # Determine the starting position based on the "type" field and comment
    type_field, commented = code_line[5:6].strip(), code_line[6:7].strip()

    start_pos = 6 
    #if len(type_field) == 1 else 7 Future opti ?
    
    # Initialize tokens dictionary
    tokens = {"type": type_field}

    # Determine Specs and Define tokens based on the type field
    if type_field == "H" and commented != "*":
        # Define tokens for H-spec
        tokens.update({
            "debug": code_line[start_pos+8:start_pos+9].strip(),
            "option_c": code_line[start_pos+11:start_pos+12].strip(),
            "option_d": code_line[start_pos+12:start_pos+13].strip(),
            "option_y": code_line[start_pos+13:start_pos+14].strip(),
            "option_n": code_line[start_pos+14:start_pos+15].strip(),
            "date_edition": code_line[start_pos+34:start_pos+35].strip(),
            "file_translation": code_line[start_pos+36:start_pos+37].strip(),
            "transparent_option": code_line[start_pos+50:start_pos+51].strip(),
            "program_id": code_line[start_pos+67:].strip(),
            "comments": code_line[start_pos+68:].strip()
        })
    elif type_field == "F" and commented != "*":
        # Define tokens for F-spec
        filename = code_line[start_pos:start_pos+8].strip()
        if filename:  # Filename is present
            tokens.update({
                "filename": filename,
                "file_type": code_line[start_pos+8:start_pos+9].strip(),
                "designation": code_line[start_pos+9:start_pos+10].strip(),
                "eof": code_line[start_pos+10:start_pos+11].strip(),
                "sequence": code_line[start_pos+11:start_pos+12].strip(),
                "format": code_line[start_pos+13:start_pos+14].strip(),
                "record_length": code_line[start_pos+17:start_pos+21].strip(),
                "limit": code_line[start_pos+21:start_pos+22].strip(),
                "key_lenght": code_line[start_pos+22:start_pos+23].strip(),
                "adr_type": code_line[start_pos+23:start_pos+24].strip(),
                "organization": code_line[start_pos+24:start_pos+25].strip(),
                "overflow_indicator": code_line[start_pos+26:start_pos+28].strip(),
                "key_location": code_line[start_pos+28:start_pos+32].strip(),
                "extension": code_line[start_pos+32:start_pos+33].strip(),
                "device": code_line[start_pos+33:start_pos+40].strip(),
                "continuation": code_line[start_pos+46:start_pos+47].strip(),
                "routine": code_line[start_pos+47:start_pos+53].strip(),
                "entry": code_line[start_pos+53:start_pos+59].strip(),
                "add": code_line[start_pos+59:start_pos+60].strip(),
                "condition": code_line[start_pos+64:start_pos+66].strip(),
                "comments": code_line[start_pos+68:].strip()
            })
        else:  # Filename is blank, handle extended record format
            tokens.update({
                "ext_record": code_line[start_pos+12:start_pos+22].strip(),
                "recoard_number": code_line[start_pos+40:start_pos+46].strip(),
                "key": code_line[start_pos+46:start_pos+47].strip(),
                "option": code_line[start_pos+47:start_pos+53].strip(),
                "entry": code_line[start_pos+51:start_pos+61].strip(),
                "comments": code_line[start_pos+68:].strip()
            })
    elif type_field == "E" and commented != "*":
        # Define tokens for E-spec
        tokens.update({
            "from_file": code_line[start_pos+4:start_pos+12].strip(),
            "to_file": code_line[start_pos+12:start_pos+20].strip(),
            "table_name": code_line[start_pos+20:start_pos+26].strip(),
            "number_records": code_line[start_pos+26:start_pos+29].strip(),
            "table_entry": code_line[start_pos+29:start_pos+33].strip(),
            "length_of_data": code_line[start_pos+33:start_pos+36].strip(),
            "format_of_data": code_line[start_pos+36:start_pos+37].strip(),
            "data_decimal": code_line[start_pos+37:start_pos+38].strip(),
            "data_sequence": code_line[start_pos+38:start_pos+39].strip(),            
            "other_name": code_line[start_pos+39:start_pos+52].strip(),
            "length": code_line[start_pos+52:start_pos+55].strip(),
            "format": code_line[start_pos+55:start_pos+56].strip(),
            "decimal": code_line[start_pos+56:start_pos+57].strip(),
            "sequence": code_line[start_pos+57:start_pos+58].strip(),
            "comments": code_line[start_pos+51:].strip()
        })
    elif type_field == "L" and commented != "*":
        # Define tokens for L-spec
        tokens.update({
            "filename": code_line[start_pos:start_pos+8].strip(),
            "line_number": code_line[start_pos+8:start_pos+11].strip(),
            "paper_lenght": code_line[start_pos+11:start_pos+13].strip(),
            "line_number_overflow": code_line[start_pos+13:start_pos+16].strip(),
            "line_overflow": code_line[start_pos+16:start_pos+18].strip(),
            "comments": code_line[start_pos+68:].strip()
        }) 
    elif type_field == "I" and commented != "*":
        # Define tokens for I-spec
        ds = code_line[start_pos+12:start_pos+14].strip()
        field = code_line[start_pos+1:start_pos+14].strip()
        format = code_line[start_pos+36:start_pos+37].strip()
        if ds == "DS":  # Data structure exist
            tokens.update({
                "ds_name": code_line[start_pos:start_pos+6],  
                "external": code_line[start_pos+10:start_pos+11].strip(),
                "option": code_line[start_pos+10:start_pos+11].strip(),
                "ds": ds,
                "ext_file": code_line[start_pos+15:start_pos+25].strip(),
                "occur": code_line[start_pos+25:start_pos+29].strip(),
                "length": code_line[start_pos+29:start_pos+34].strip(),
                "comments": code_line[start_pos+68:].strip()
            })
        elif format == "C":  # Named constant 
            tokens.update({
                "constant": code_line[start_pos+14:start_pos+28].strip(),
                "constant_value": code_line[start_pos+36:start_pos+80].strip(),
                "constant_name": code_line[start_pos+46:start_pos+52].strip(), 
                "comments": code_line[start_pos+68:].strip()           
            })  
        elif field == "" and format != "C":  # other fields type
            tokens.update({
                "init": code_line[start_pos+1:start_pos+2].strip(),
                "ext_field_name": code_line[start_pos+14:start_pos+24],
                "data_format": format,
                "from_position": code_line[start_pos+37:start_pos+41].strip(),
                "to_position": code_line[start_pos+41:start_pos+45].strip(),
                "decimal_precision": code_line[start_pos+45:start_pos+46].strip(),
                "field_name": code_line[start_pos+46:start_pos+52].strip(),
                "control_level": code_line[start_pos+52:start_pos+54].strip(),
                "matching_field": code_line[start_pos+54:start_pos+56].strip(),
                "relation_field": code_line[start_pos+56:start_pos+58].strip(),
                "positive_field": code_line[start_pos+58:start_pos+60].strip(),
                "negative_field": code_line[start_pos+60:start_pos+62].strip(),
                "zero_blank": code_line[start_pos+62:start_pos+64].strip(),
                "comments": code_line[start_pos+68:].strip()
            })
        else:
            tokens.update({
                "filename": code_line[start_pos:start_pos+8].strip(),  
                "sequence": code_line[start_pos+8:start_pos+10].strip(),
                "number": code_line[start_pos+10:start_pos+11].strip(),  
                "record_id": code_line[start_pos+12:start_pos+14].strip(),
                "first_position": code_line[start_pos+14:start_pos+18].strip(),
                "first_non": code_line[start_pos+18:start_pos+19].strip(),
                "first_code_part": code_line[start_pos+19:start_pos+20].strip(),
                "first_char": code_line[start_pos+20:start_pos+21].strip(),
                "second_position": code_line[start_pos+21:start_pos+25].strip(),
                "second_non": code_line[start_pos+25:start_pos+26].strip(),
                "second_code_part": code_line[start_pos+26:start_pos+27].strip(),
                "second_char": code_line[start_pos+27:start_pos+28].strip(),
                "third_position": code_line[start_pos+28:start_pos+32].strip(),
                "third_non": code_line[start_pos+32:start_pos+33].strip(),
                "third_code_part": code_line[start_pos+33:start_pos+34].strip(),
                "third_char": code_line[start_pos+34:start_pos+35].strip(),
                "comments": code_line[start_pos+68:].strip()
            })                
    elif type_field == "C" and commented != "*":
        # Define tokens for C-spec
        tokens.update({
            "control": code_line[start_pos:start_pos+2].strip(),
            "indicator1": code_line[start_pos+2:start_pos+5].strip(),
            "indicator2": code_line[start_pos+5:start_pos+8].strip(),
            "indicator3": code_line[start_pos+8:start_pos+11].strip(),
            "factor1": code_line[start_pos+11:start_pos+21].strip(),
            "opcode": code_line[start_pos+21:start_pos+26].strip(),
            "factor2": code_line[start_pos+26:start_pos+36].strip(),
            "result": code_line[start_pos+36:start_pos+42].strip(),
            "len": code_line[start_pos+42:start_pos+45].strip(),
            "de": code_line[start_pos+45:start_pos+47].strip(),
            "hi": code_line[start_pos+47:start_pos+49].strip(),
            "lo": code_line[start_pos+49:start_pos+51].strip(),
            "eq": code_line[start_pos+51:start_pos+53].strip(),
            "comments": code_line[start_pos+53:].strip()
        })
    elif type_field == "O" and commented != "*":
        add_del = code_line[start_pos+9:start_pos+12].strip()
        and_or = code_line[start_pos+7:start_pos+10].strip()
        named = code_line[start_pos+1:start_pos+16].strip()
        # Define tokens for O-spec
        # Disk output
        if add_del == "ADD" or add_del == "DEL":
            tokens.update({
                "name": code_line[start_pos:start_pos+8].strip(),
                "type": code_line[start_pos+8:start_pos+9].strip(),
                "add_del": add_del,
                "indicator1": code_line[start_pos+16:start_pos+19].strip(),
                "indicator2": code_line[start_pos+19:start_pos+22].strip(),
                "indicator3": code_line[start_pos+22:start_pos+25].strip(),
                "excpt_name": code_line[start_pos+25:start_pos+31].strip(),
                "comments": code_line[start_pos+68:].strip()
            })
        # Additional record 
        elif and_or == "AND" or and_or == "OR":
            tokens.update({
                "and_or": and_or,
                "before_space": code_line[start_pos+10:start_pos+11].strip(),
                "after_space": code_line[start_pos+11:start_pos+12].strip(),
                "before_skip": code_line[start_pos+12:start_pos+14].strip(),
                "after_skip": code_line[start_pos+14:start_pos+16].strip(),
                "indicator1": code_line[start_pos+16:start_pos+19].strip(),
                "indicator2": code_line[start_pos+19:start_pos+22].strip(),
                "indicator3": code_line[start_pos+22:start_pos+25].strip(),
                "excpt_name": code_line[start_pos+25:start_pos+31].strip(),
                "comments": code_line[start_pos+68:].strip()
            })
        # Field output    
        elif not named:
            tokens.update({
                "indicator1": code_line[start_pos+16:start_pos+19].strip(),
                "indicator2": code_line[start_pos+19:start_pos+22].strip(),
                "indicator3": code_line[start_pos+22:start_pos+25].strip(),
                "field_name": code_line[start_pos+25:start_pos+31].strip(),
                "editcode": code_line[start_pos+31:start_pos+32].strip(),
                "after_blank": code_line[start_pos+32:start_pos+33].strip(),
                "end_position": code_line[start_pos+33:start_pos+37].strip(),
                "type": code_line[start_pos+37:start_pos+38].strip(),
                "editword_constant": code_line[start_pos+38:start_pos+64].strip(),
                "comments": code_line[start_pos+68:].strip()
            })
        # Record output     
        else:
            tokens.update({
                "name": code_line[start_pos:start_pos+8].strip(),
                "type": code_line[start_pos+8:start_pos+9].strip(),
                "anticipated_call": code_line[start_pos+9:start_pos+10].strip(),
                "before_space": code_line[start_pos+10:start_pos+11].strip(),
                "after_space": code_line[start_pos+11:start_pos+12].strip(),
                "before_skip": code_line[start_pos+12:start_pos+14].strip(),
                "after_skip": code_line[start_pos+14:start_pos+16].strip(),
                "indicator1": code_line[start_pos+16:start_pos+19].strip(),
                "indicator2": code_line[start_pos+19:start_pos+22].strip(),
                "indicator3": code_line[start_pos+22:start_pos+25].strip(),
                "excpt_name": code_line[start_pos+25:start_pos+31].strip(),
                "comments": code_line[start_pos+68:].strip()
            })
    else:        
        tokens.update({
            "comments": code_line[start_pos+2:].strip()
            })                                                               
    return tokens

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