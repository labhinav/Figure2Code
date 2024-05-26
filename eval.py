import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os  # Import the os module
from codebleu import calc_codebleu
from datasets import load_dataset

def min_l1_distance_with_padding(list1, list2):
    m, n = len(list1), len(list2)
    
    # Initialize the dp table
    dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0
    
    # Initialize the first row and first column
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] + abs(list1[i-1] - 0)
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] + abs(0 - list2[j-1])
    
    # Fill the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(dp[i-1][j-1] + abs(list1[i-1] - list2[j-1]),
                           dp[i-1][j] + abs(list1[i-1] - 0),
                           dp[i][j-1] + abs(0 - list2[j-1]))
    
    ans = dp[m][n]
    #normalize the answer by the maximum value in the lists, take into account that list may be empty
    max_val = max(max(list1, default=0), max(list2, default=0))
    if max_val == 0:
        return ans
    return ans/max_val

#write a function to get average min padded l1 between two list of lists
def get_l1(preds,format_string_pred):
    total_l1=0
    failed_files = []
    for i in range(len(preds)):
        #apply the index to the format string to get the target file
        # print(i,preds[i])
        target_file = format_string_pred.format(i)
        #read a list of floats from the file. each line is a float
        with open(target_file, 'r') as f:
            target_values = f.readlines()
            #split the string into a list of floats
            try:
                target_values = [float(x) for x in target_values]
            except ValueError:
                failed_files.append(target_file)
                continue
        pred_values = preds[i]
        pred_values = [float(x) for x in pred_values[1:-1].split(',')]
        l1 = min_l1_distance_with_padding(pred_values, target_values)
        print(l1)
        total_l1+=l1
    average_l1 = total_l1/len(preds)
    return average_l1, failed_files

def get_image_mse(preds,targets):
    total_mse=0
    for i in range(len(preds)):
        mse = np.mean((preds[i] - targets[i])**2)
        total_mse+=mse
    average_mse = total_mse/len(preds)
    return average_mse

def batch_process_images(df, format_string_pred, batch_size=10):
    """Process images in batches and calculate MSE.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the images column.
        format_string_pred (str): Format string for predicted images path.
        batch_size (int): Number of images to process in each batch.

    Returns:
        float: Average MSE over all batches.
    """
    num_images = len(df)
    batch_mse = []
    batch_sizes = []
    for start in range(0, num_images, batch_size):
        end = min(start + batch_size, num_images)  # Ensure we don't go out of bounds
        targets = []
        preds = []
        for i in range(start, end):
            # Get the target image from the DataFrame
            target_file_name = df['og_file_name'][i]

            # Construct the predicted image file name using the format string and index
            pred_file_name = format_string_pred.format(i)

            # Check if pred_image path exists
            if not os.path.exists(pred_file_name):
                continue

            # Read the target image from images/ folder
            target_image = plt.imread(os.path.join('images', target_file_name))
            pred_image = plt.imread(pred_file_name)

            targets.append(target_image)
            preds.append(pred_image)
        batch_sizes.append(len(preds))

        # Calculate MSE for the current batch
        mse = get_image_mse(preds, targets)
        batch_mse.append(mse)
        print({'start': start, 'end': end, 'mse': mse})
    #calculate the average mse weighted by the batch sizes
    weighted_mse = np.average(batch_mse, weights=batch_sizes)
    return weighted_mse


def get_codebleu_scores(df, format_string_pred):
    total_codebleu=0
    for index in range(len(df)):
        target_file = format_string_pred.format(index)
        #read code from the file
        with open(target_file, 'r') as f:
            target_code = f.read()
        codebleu_score = calc_codebleu([df['code'][index]], [target_code], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
        total_codebleu+=codebleu_score['codebleu']
        # print(codebleu_score['codebleu'])
    return total_codebleu/len(df)

# Load a dataset (e.g., 'squad', 'glue', etc.)
dataset = load_dataset('abhinavl/figure2code_data')
#access the test split
df = dataset['test']
pred_values = df['values']
average_padding_l1, failed_files = get_l1(pred_values, 'inference_output_values/test_{}.txt')
print("Average Padding L1 Distance:", average_padding_l1)
print("Failed percentage:", len(failed_files)/len(pred_values))
print("Failed files:", failed_files)