import ast
import os
import numpy as np

def extract_values_from_file(file_path, search_string='values'):
    values = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip().startswith(search_string):
                # Extract the part after 'values ='
                values_str = line.split('=')[1].strip()
                if ']' not in values_str:
                    values_str += ']'
                # Check if the values are of type []
                if '[' in values_str and 'np.array' not in values_str:
                    # Use ast.literal_eval to safely evaluate the list
                    values = ast.literal_eval(values_str)
                # Check if the values are of type np.array([])
                elif 'np.array' in values_str:
                    # Extract the part inside the np.array()
                    values_str = values_str.split('np.array(')[1].split(')')[0]
                break
    return values

folder_path = 'llava_fine_tuned/filtered_code_challenge'  
search_string = 'values'
# Create a new folder to save the values
output_folder_path = 'llava_fine_tuned/challenge_values' 
os.makedirs(output_folder_path, exist_ok=True)
error_files = []
# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    print(f"Processing file: {file_path}")

    try:
        values = extract_values_from_file(file_path, search_string)
    except Exception as e:
        error_files.append(filename)

    # Create the new file path with the same name but with .txt extension
    new_filename = os.path.splitext(filename)[0] + '.txt'
    new_file_path = os.path.join(output_folder_path, new_filename)

    # Save the values to the new file
    with open(new_file_path, 'w') as new_file:
        for value in values:
            new_file.write(str(value) + '\n')

    print(f"Values saved to {new_file_path}")
print(f"Error files: {error_files}")
print(f"No. of error files: {len(error_files)}")