import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from codebleu import calc_codebleu

def get_image_mse(preds,targets):
    total_mse=0
    for i in range(len(preds)):
        mse = np.mean((preds[i] - targets[i])**2)
        total_mse+=mse
    average_mse = total_mse/len(preds)
    return average_mse

def batch_process_images(file_names, batch_size=10):
    """Process images in batches and calculate MSE."""
    num_images = len(file_names)
    batch_mse = []
    for start in range(0, num_images, batch_size):
        end = start + batch_size
        targets = []
        preds = []
        for file_name in file_names[start:end]:
            image = plt.imread(f'images/{file_name}')
            targets.append(image)
            # add a random prediction
            preds.append(np.random.rand(*image.shape))
        mse = get_image_mse(preds, targets)
        batch_mse.append(mse)
        print({'start': start, 'end': end, 'mse': mse})
    return np.mean(batch_mse)

def get_codebleu_scores(df):
    total_codebleu=0
    random_code = """
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
x = np.random.rand(50)
y = 2 * x + np.random.normal(0, 0.1, 50)  # y = 2x + noise

# Create a scatter plot
plt.scatter(x, y, color='blue', label='Data points')

# Fit a line to the random data
m, b = np.polyfit(x, y, 1)  # Fit a 1-degree polynomial (line) to the data
plt.plot(x, m*x + b, color='red', label=f'Fitted line: y={m:.2f}x+{b:.2f}')

# Adding labels and title
plt.xlabel('Random X values')
plt.ylabel('Random Y values')
plt.title('Random Scatter Plot with Fitted Line')
plt.legend()

# Show the plot
plt.show()
"""
    for index, row in df.iterrows():
        codebleu_score = calc_codebleu([row['code']], [random_code], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
        total_codebleu+=codebleu_score['codebleu']
        # print(codebleu_score['codebleu'])
    return total_codebleu/len(df)

# Read the CSV file
df = pd.read_csv('images/metadata.csv')

# Process images in batches
# final_mse = batch_process_images(df['file_name'].tolist())
# print(final_mse)
print(get_codebleu_scores(df))
