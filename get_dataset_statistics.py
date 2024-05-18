import pandas as pd

import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('metadata/train_metadata.csv')

# Extract the labels column
labels = data['labels']

# Count the frequencies of the number of labels per sample
label_counts = labels.str.split(',').apply(len).value_counts().sort_index()

# Generate the chart
plt.bar(label_counts.index, label_counts.values)
plt.xlabel('Number of Labels')
plt.ylabel('Frequency')
plt.title('Frequencies of Number of Labels per Sample')

# Save the chart
plt.savefig('label_frequencies.png')
