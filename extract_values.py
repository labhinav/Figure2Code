import ast
import os

def extract_values_from_file(file_path):
    values = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip().startswith('categories='):
                # Extract the part after 'values ='
                values_str = line.split('=')[1].strip()
                if ']' not in values_str:
                    values_str += ']'
                # Use ast.literal_eval to safely evaluate the list
                print(values_str)
                values = ast.literal_eval(values_str)
                break
    return values

folder_path = 'inference_output_new'  

# Create a new folder to save the values
output_folder_path = 'inference_output_new_categories' 
os.makedirs(output_folder_path, exist_ok=True)

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    print(f"Processing file: {file_path}")
    values = extract_values_from_file(file_path)

    # Create the new file path with the same name but with .txt extension
    new_filename = os.path.splitext(filename)[0] + '.txt'
    new_file_path = os.path.join(output_folder_path, new_filename)

    # Save the values to the new file
    with open(new_file_path, 'w') as new_file:
        for value in values:
            new_file.write(str(value) + '\n')

    print(f"Values saved to {new_file_path}")