
# RPG III Code Autoencoder

## Overview

This repository contains a Python script designed to process RPG III code, tokenize it, and train an autoencoder using a Transformer-based neural network. The script uses TensorFlow and Keras for building the model and includes functionality for loading datasets, tokenizing text, training the model, and evaluating its performance.

## Features

- Load and preprocess RPG III code datasets.
- Tokenize RPG III code lines.
- Train a Transformer-based autoencoder model.
- Evaluate model performance using various metrics.
- Save and load trained models.
- Generate and visualize performance reports.

## Installation

To use the script, you need to have Python installed on your machine along with the required dependencies.

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/rpgiii-autoencoder.git
    cd rpgiii-autoencoder
    ```

2. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

The script can be run directly from the command line. Below are the main functionalities and how to use them:

### 1. Tokenize RPG III Code

The `tokenize_rpgiii_code` function, located in `ModelTesting/Parser/tokenizer.py`, tokenizes a line of RPG III code and returns a dictionary of tokens.

#### Tokenizer Usage

```python
from Parser.tokenizer import tokenize_rpgiii_code

# Example RPG III code line
rpgiii_line = "     C                     Z-ADD*ZEROS    NOAFF"
              
# Tokenize the RPG III code line
rpgiii_tokens = tokenize_rpgiii_code(rpgiii_line)

print(f'Tokens: {rpgiii_tokens}')
```

#### Tokenizer Method

- `tokenize_rpgiii_code(code_line)`: Tokenizes a line of RPG III code and returns a dictionary of tokens.

### 2. Define Transformer Layers

Define the Transformer Encoder and Decoder layers.

```python
def transformer_encoder(vocab_size, embed_dim, num_heads, ff_dim, max_sequence_len, rate=0.1):
    ...
    return Model(inputs, transformer_block)

def transformer_decoder(vocab_size, embed_dim, num_heads, ff_dim, max_sequence_len, rate=0.1):
    ...
    return Model(inputs, outputs)
```

### 3. Process Files and Prepare Data

Load and preprocess RPG III code files.

```python
import pandas as pd
import os
import glob
import concurrent.futures

def process_file(filepath):
    df = pd.read_csv(filepath, names=['sentence'], sep='\t', usecols=[0])
    df['tokens'] = df['sentence'].apply(lambda x: list(tokenize_rpgiii_code(x).values()))
    return df

file_directory = '/path/to/your/files'
file_pattern = os.path.join(file_directory, '*.txt')

dataframes = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_file, filepath): filepath for filepath in glob.glob(file_pattern)}
    for future in concurrent.futures.as_completed(futures):
        dataframes.append(future.result())

combined_df = pd.concat(dataframes, ignore_index=True)
```

### 4. Train the Autoencoder

Train the autoencoder model with the preprocessed data.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

# Tokenize and pad sequences
tokenizer = Tokenizer(filters='', oov_token='<OOV>')
all_tokens = [token for sublist in combined_df['tokens'].tolist() for token in sublist]
tokenizer.fit_on_texts(all_tokens)
sequences = tokenizer.texts_to_sequences(combined_df['tokens'].tolist())
max_sequence_len = max([len(seq) for seq in sequences])
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

data = pad_sequences(sequences, maxlen=max_sequence_len, padding='post')

# Define hyperparameters
embed_dim = 128  # Embedding size for each token
num_heads = 5    # Number of attention heads
ff_dim = 128     # Hidden layer size in feed forward network inside transformer
batch_size = 20 
epoch = 32

# Define and compile the autoencoder model
encoder = transformer_encoder(vocab_size, embed_dim, num_heads, ff_dim, max_sequence_len)
decoder = transformer_decoder(vocab_size, embed_dim, num_heads, ff_dim, max_sequence_len)

autoencoder_input = Input(shape=(max_sequence_len,))
encoded_seq = encoder(autoencoder_input)
decoded_seq = decoder(encoded_seq)

autoencoder = Model(autoencoder_input, decoded_seq)
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',
                    metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.metrics.Precision(), tf.metrics.Recall()])

# Prepare target data
target_data = tf.keras.utils.to_categorical(data, vocab_size)

# Enable mixed precision training
mixed_precision = tf.keras.mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Split the data
train_data, test_data, train_labels, test_labels = train_test_split(data, target_data, test_size=0.2, random_state=42)

# Train the model
history = autoencoder.fit(train_data, train_labels, epochs=epoch, batch_size=batch_size)
```

### 5. Evaluate the Model

Evaluate the model on the test data and visualize the performance.

```python
import matplotlib.pyplot as plt

# Evaluate the model
test_results = autoencoder.evaluate(test_data, test_labels)
print(f'Test Results: {test_results}')

# Plot the training loss
plt.plot(history.history['loss'], label='Training loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

### 6. Save and Load the Model

Save the trained model and load it for future use.

```python
# Save the entire model
autoencoder.save('autoencoder_model.h5')

# Load the model
loaded_model = tf.keras.models.load_model('autoencoder_model.h5')
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project is maintained by [ArmonieIA](https://github.com/ArmonieIA). Special thanks to all the contributors who have helped improve the project.

---

Feel free to modify this `README.md` to suit your specific requirements or to add more details about your project.
