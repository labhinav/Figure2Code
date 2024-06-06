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



# Load the DataFrame from a CSV file
df = pd.read_csv('results/generated_codes_new.csv')

# Specify the column where the replacement should occur
column_name = 'Generated_Code'

# Replace all occurrences of '< n >' with '\n' in the specified column
df[column_name] = df[column_name].str.replace('< n >', '\n')
# Apply the clean_code function to the specified column
df[column_name] = df[column_name].apply(clean_code)
# Optionally, save the modified DataFrame back to a CSV file
df.to_csv('results/generated_codes_new_replaced.csv', index=False)

# Display the modified DataFrame
print(df)