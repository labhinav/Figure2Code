import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os  # Import the os module
from codebleu import calc_codebleu
from datasets import load_dataset

def get_image_mse(preds,targets):
    total_mse=0
    for i in range(len(preds)):
        mse = np.mean((preds[i] - targets[i])**2)
        total_mse+=mse
    average_mse = total_mse/len(preds)
    return average_mse

def get_l1(preds,targets):
    total_l1=0
    for i in range(len(preds)):
        #split the string into a list of floats
        preds[i] = [float(x) for x in preds[i].split(',')]
        targets[i] = [float(x) for x in targets[i].split(',')]
        l1 = np.mean(np.abs(preds[i] - targets[i]))
        total_l1+=l1
    average_l1 = total_l1/len(preds)
    return average_l1

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
# Process images in batches
final_mse = batch_process_images(df, 'inferenced_output_processed_images/test_{:d}.py.png', batch_size=10)
print(final_mse)
print(get_codebleu_scores(df, 'inference_output_processed/test_{:d}.py'))

# print(get_l1(df['values'],df['values']))
