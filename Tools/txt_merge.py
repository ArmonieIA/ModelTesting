import os

# Directory where the .txt files are located
directory = 'Python/DeepLearningTrain/sample_data/RPG3_convert/DSSSRC'

# Name of the output file
output_filename = 'merged_file.txt'

# Get a list of .txt files in the directory
txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]

# Open the output file in write mode with utf-8 encoding
with open(output_filename, 'w', encoding='utf-8') as outfile:
    # Iterate over each file
    for fname in txt_files:
        # Open each file in read mode with utf-8 encoding
        with open(os.path.join(directory, fname), 'r', encoding='utf-8') as infile:
            # Read the content of the file and write it to the output file
            outfile.write(infile.read())
            # Optionally, write a newline after each file's content
            outfile.write('\n')

print(f"All .txt files have been merged into {output_filename}")