import os

folder_path = 'inference_output'  # Replace with the actual folder path
output_folder_path = 'inference_output_processed'  # Replace with the path of the new folder

for filename in os.listdir(folder_path):
    if filename.endswith('.py'):  # Process only Python files
        file_path = os.path.join(folder_path, filename)
        output_file_path = os.path.join(output_folder_path, filename)  # Create the output file path

        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Remove starting <s> tags
        lines = [line.replace('<s>', '') for line in lines]

        #Remove ending </s> tags
        lines = [line.replace('</s>', '') for line in lines]

        # Find the index of the first import statement
        import_index = next((i for i, line in enumerate(lines) if line.startswith('import') or line.startswith('from')), None)

        if import_index is not None:
            # Remove all text before the first import statement
            lines = lines[import_index:]

        # Find the index of plt.show()
        show_index = next((i for i, line in enumerate(lines) if 'plt.show()' in line), None)

        if show_index is not None:
            # Remove all text after plt.show()
            lines = lines[:show_index+1]

        #filter ending and trailing whitespaces
        lines = [line.strip() for line in lines]

        with open(output_file_path, 'w') as file:  # Write to the output file path
            file.write('\n'.join(lines))  # Write the lines with original spacing
