import pandas as pd

# Load the train_metadata_with_code.csv file
df = pd.read_csv('images/metadata.csv')

# Add a file_name column with value figure_{index}.png at each row
df['og_file_name'] = [f'figure_{i}.png' for i in range(len(df))]

# Save the modified DataFrame to images/metadata.csv
df.to_csv('images/metadata.csv', index=False)