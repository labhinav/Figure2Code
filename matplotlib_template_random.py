import pandas as pd
import random

def random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

import random

def random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

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
    # List of possible fonts
    fonts = ['Arial', 'Verdana', 'Times New Roman', 'Comic Sans MS', 'Courier New']

    # Random font selection
    font = random.choice(fonts)
    # Generating a single random color for all bars
    color = random_color()

    code = f"""
import matplotlib.pyplot as plt

# Categories and their corresponding values
categories = {categories}
values = {values}

# Creating the bar chart
plt.figure(figsize=(8, 5))  # Set the figure size (optional)
plt.bar(categories, values, color='{color}')  # Plot the bars with a random color

# Adding title and labels
plt.title('{title}', fontname='{font}')  # Add a title to the chart
plt.xlabel('{xlabel}', fontname='{font}')  # Label for the x-axis
plt.ylabel('{ylabel}', fontname='{font}')  # Label for the y-axis

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
    # List of possible fonts
    fonts = ['Arial', 'Verdana', 'Times New Roman', 'Comic Sans MS', 'Courier New']

    # Random font selection
    font = random.choice(fonts)
    # Generating a single random color for scatter points
    color = random_color()

    code = f"""
import matplotlib.pyplot as plt

# Categories and their corresponding values
categories = {categories}
values = {values}

# Creating the scatter plot
plt.figure(figsize=(8, 5))  # Set the figure size (optional)
plt.scatter(categories, values, color='{color}')  # Plot the scatter points with a random color

# Adding title and labels
plt.title('{title}', fontname='{font}')  # Add a title to the chart
plt.xlabel('{xlabel}', fontname='{font}')  # Label for the x-axis
plt.ylabel('{ylabel}', fontname='{font}')  # Label for the y-axis

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
    # List of possible fonts
    fonts = ['Arial', 'Verdana', 'Times New Roman', 'Comic Sans MS', 'Courier New']

    # Random font selection
    font = random.choice(fonts)
    #convert values to a list of integers from a string such as '[1,2,3]'
    values = values.replace('[','').replace(']','').split(',')
    values = [float(value) for value in values]
    #multiply all negative numbers by -1
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
plt.pie(values, labels=categories, autopct='%1.1f%%', colors=plt.cm.Set3.colors)  # Plot the pie chart with percentages and a color map

# Adding title and labels
plt.title('{title}', fontname='{font}')  # Add a title to the chart

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
df3 = df3.sample(1000)
df3['code'] = df3.apply(apply_code_pie, axis=1)
# Apply the function and create a new column
df = pd.read_csv('metadata/train_metadata_with_code.csv')
#Randomly select 1000 rows
df = df.sample(1000)
df['code'] = df.apply(apply_code_bar, axis=1)

df2 = pd.read_csv('metadata/train_metadata_with_code.csv')
df2 = df2.sample(1000)
df2['code'] = df2.apply(apply_code_scatter, axis=1)

#concatenate the dataframes
df = pd.concat([df, df2, df3])

#save the df to a new csv file
df.to_csv('metadata/train_metadata_with_code_challenge.csv', index=False)

