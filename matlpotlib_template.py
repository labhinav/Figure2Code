import pandas as pd
def generate_matplotlib_code(categories, values, xlabel, ylabel, title):
    """
    Generates a string containing Python code to create a bar chart using Matplotlib.
    
    Parameters:
        categories (list): The categories to be used on the x-axis.
        values (list): The values for each category.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the chart.
        
    Returns:
        str: Python code as a string.
    """
    code = f"""
import matplotlib.pyplot as plt

# Categories and their corresponding values
categories = {categories}
values = {values}

# Creating the bar chart
plt.figure(figsize=(8, 5))  # Set the figure size (optional)
plt.bar(categories, values, color='skyblue')  # Plot the bars with skyblue color

# Adding title and labels
plt.title('{title}')  # Add a title to the chart
plt.xlabel('{xlabel}')  # Label for the x-axis
plt.ylabel('{ylabel}')  # Label for the y-axis

# Display the chart
plt.show()
"""
    return code

def apply_code(row):
    return generate_matplotlib_code(row['labels'], row['values'], 'Categories', row['value_heading'], row['title'])

# Apply the function and create a new column
df = pd.read_csv('metadata/train_metadata.csv')
df['code'] = df.apply(apply_code, axis=1)

#save the df to a new csv file
df.to_csv('metadata/train_metadata_with_code.csv', index=False)

