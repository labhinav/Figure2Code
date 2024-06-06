import pandas as pd

def generate_matplotlib_code_bar(categories, values, xlabel, ylabel, title):
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
plt.bar(categories, values, color='skyblue')  # Plot the bars with a random color

# Adding title and labels
plt.title('{title}')  # Add a title to the chart
plt.xlabel('{xlabel}')  # Label for the x-axis
plt.ylabel('{ylabel}')  # Label for the y-axis

# Display the chart
plt.show()
"""
    return code

def generate_matplotlib_code_scatter(categories, values, xlabel, ylabel, title):
    """
    Generates a string containing Python code to create a scatter plot using Matplotlib.
    
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

# Creating the scatter plot
plt.figure(figsize=(8, 5))  # Set the figure size (optional)
plt.scatter(categories, values, color='skyblue')  # Plot the scatter points with a random color

# Adding title and labels
plt.title('{title}')  # Add a title to the chart
plt.xlabel('{xlabel}')  # Label for the x-axis
plt.ylabel('{ylabel}')  # Label for the y-axis

# Display the chart
plt.show()
"""
    return code

def generate_matplotlib_code_pie(categories, values, xlabel, ylabel, title):
    """
    Generates a string containing Python code to create a pie chart using Matplotlib.
    
    Parameters:
        categories (list): The categories to be used on the x-axis.
        values (list): The values for each category.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the chart.
        
    Returns:
        str: Python code as a string.
    """
    values = values.replace('[','').replace(']','').split(',')
    values = [float(value) for value in values]
    values = [value if value >= 0 else -1*value for value in values]
    if max(values) == 0:
        values = [1 for value in values]
    code = f"""
import matplotlib.pyplot as plt

# Categories and their corresponding values
categories = {categories}
values = {values}

# Creating the pie chart
plt.figure(figsize=(8, 5))  # Set the figure size (optional)
plt.pie(values, labels=categories, autopct='%1.1f%%', colors=plt.cm.Paired.colors)  # Plot the pie chart with percentages and a color map

# Adding title and labels
plt.title('{title}')  # Add a title to the chart

# Display the chart
plt.show()
"""
    return code


def apply_code_bar(row):
    return generate_matplotlib_code_bar(row['labels'], row['values'], 'Categories', row['value_heading'], row['title'])

def apply_code_scatter(row):
    return generate_matplotlib_code_scatter(row['labels'], row['values'], 'Categories', row['value_heading'], row['title'])

def apply_code_pie(row):
    return generate_matplotlib_code_pie(row['labels'], row['values'], 'Categories', row['value_heading'], row['title'])

#do the same for pie chart
df3 = pd.read_csv('metadata/train_metadata_with_code.csv')
df3['code'] = df3.apply(apply_code_pie, axis=1)
# Apply the function and create a new column
df = pd.read_csv('metadata/train_metadata_with_code.csv')
df['code'] = df.apply(apply_code_bar, axis=1)

df2 = pd.read_csv('metadata/train_metadata_with_code.csv')
df2['code'] = df2.apply(apply_code_scatter, axis=1)

#concatenate the dataframes
df = pd.concat([df, df2, df3])

#save the df to a new csv file
df.to_csv('metadata/train_metadata_with_code_new.csv', index=False)

