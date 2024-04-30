import pandas as pd

filepath_dict = {'pgm': 'Python/DeepLearningTrain/sample_data/contexte.txt'}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)

df = pd.concat(df_list)

# Iterate over each row in the DataFrame and print it
for index, row in df.iterrows():
    print(row)