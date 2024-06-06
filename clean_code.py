import os
import pandas as pd
import re

# Function to clean code in the column by removing spaces before or after any non-alphanumeric character
def clean_code(code):
    # Split the code into lines
    lines = code.split('\n')
    # Clean each line individually
    cleaned_lines = [re.sub(r'(\s*)([^\w\s])(\s*)', r'\2', line) for line in lines]
    # Combine the cleaned lines back into a single string
    cleaned_code = '\n'.join(cleaned_lines)
    return cleaned_code

# Specify the input folder and output folder paths
input_folder = 'llava_baseline/filtered_code_new'
output_folder = 'llava_baseline/cleaned_code_new'

# Iterate over the files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.py'):  # Assuming the files are Python files
        # Load the code from the Python file
        input_file = os.path.join(input_folder, filename)
        with open(input_file, 'r') as file:
            code = file.read()
        
        # Clean the code using the clean_code function
        cleaned_code = clean_code(code)
        
        # Create the output file path
        output_file = os.path.join(output_folder, filename)
        
        # Save the modified code to a new Python file in the output folder
        with open(output_file, 'w') as file:
            file.write(cleaned_code)
