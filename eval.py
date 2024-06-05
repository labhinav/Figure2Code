import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os  # Import the os module
from codebleu import calc_codebleu
from datasets import load_dataset
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu


def min_l1_distance_with_padding(list1, list2):
    max_val = max(list2, default=0)
    m, n = len(list1), len(list2)
    #swap the lists if the first list is longer
    if m > n:
        list1, list2 = list2, list1
        m, n = n, m
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
                           dp[i-1][j] + abs(list1[i-1] - 0))
    
    ans = dp[m][n]
    #normalize the answer by the maximum value in the lists, take into account that list may be empty
    if max_val == 0:
        normalized_ans = ans
    else:
        normalized_ans = ans/max_val
    #clip the answer by max(m,n)
    normalized_ans = min(normalized_ans, max(m,n))
    return normalized_ans

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
        # print(l1)
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

def batch_process_images(df, format_string_pred, image_folder, batch_size=10):
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
            target_image = plt.imread(os.path.join(image_folder, target_file_name))
            pred_image = plt.imread(pred_file_name)

            targets.append(target_image)
            preds.append(pred_image)
        batch_sizes.append(len(preds))

        # Calculate MSE for the current batch
        mse = get_image_mse(preds, targets)
        batch_mse.append(mse)
        # print({'start': start, 'end': end, 'mse': mse})
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

def get_codebleu_scores_from_df(df_preds, df_targets):
    total_codebleu=0
    for index in range(len(df_preds)):
        codebleu_score = calc_codebleu([df_preds['code'][index]], [df_targets['Generated_Code'][index]], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
        total_codebleu+=codebleu_score['codebleu']
        # print(codebleu_score['codebleu'])
    return total_codebleu/len(df_preds)



def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    # print(f"List 1: {list1}")
    # print(f"List 2: {list2}")
    # print(f"Intersection: {intersection}")
    # print(f"Union: {union}")
    return intersection / union if union != 0 else 0

def get_average_jaccard(preds, format_string_pred):
    total_jaccard = 0
    failed_files = []
    for i in range(len(preds)):
        target_file = format_string_pred.format(i)
        try:
            with open(target_file, 'r') as f:
                target_values = f.readlines()
                target_values = [x.strip() for x in target_values]
        except (ValueError, FileNotFoundError):
            failed_files.append(target_file)
            continue

        try:
            pred_values = [x.strip() for x in preds[i][1:-1].split(',')]
        except ValueError:
            failed_files.append(target_file)
            continue

        jaccard = jaccard_similarity(pred_values, target_values)
        total_jaccard += jaccard

    average_jaccard = total_jaccard / len(preds)
    return average_jaccard, failed_files

def convert_to_black_and_white(image):
    grayscale_image = rgb2gray(image)
    thresh = threshold_otsu(grayscale_image)
    binary_image = grayscale_image > thresh
    return binary_image.astype(np.float64)

def batch_process_images_bw(df, format_string_pred, image_folder, batch_size=10):
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
            target_image = plt.imread(os.path.join(image_folder, target_file_name))
            pred_image = plt.imread(pred_file_name)

            # Convert images to black and white
            target_image_bw = convert_to_black_and_white(target_image)
            pred_image_bw = convert_to_black_and_white(pred_image)

            targets.append(target_image_bw)
            preds.append(pred_image_bw)
        batch_sizes.append(len(preds))

        # Calculate MSE for the current batch
        mse = get_image_mse(preds, targets)
        batch_mse.append(mse)
        print({'start': start, 'end': end, 'mse': mse})
    
    # Calculate the average mse weighted by the batch sizes
    weighted_mse = np.average(batch_mse, weights=batch_sizes)
    return weighted_mse

def extract_chart_type(code):
    if 'plt.bar' in code:
        return 'bar'
    elif 'plt.scatter' in code:
        return 'scatter'
    elif 'plt.pie' in code:
        return 'pie'
    else:
        return 'other'

def get_chart_accuracy(df, format_string_pred):
    correct_count = 0
    for index in range(len(df)):
        target_file = format_string_pred.format(index)
        # read code from the file
        with open(target_file, 'r') as f:
            target_code = f.read()

        pred_code = df['code'][index]

        target_chart_type = extract_chart_type(target_code)
        pred_chart_type = extract_chart_type(pred_code)
        # print("Predicted Type",target_chart_type)
        # print("Target Type",pred_chart_type)
        if target_chart_type == pred_chart_type:
            correct_count += 1

    accuracy = correct_count / len(df)
    return accuracy

# Load a dataset (e.g., 'squad', 'glue', etc.)
dataset = load_dataset('abhinavl/figure2code_new_data_square', split='train')
#shuffle the dataset
df = dataset
# print(df)
# print(dataset[0])
# print(df[0])
#access the test split
df = dataset[0:100]
# print(type(df))
format_string_pred = 'llava_fine_tuned/filtered_code_new/test_{}.py'
codebleu_score = get_codebleu_scores(df, format_string_pred)
print("CodeBLEU Score:", codebleu_score)
chart_accuracy = get_chart_accuracy(df, format_string_pred)
print("Chart Accuracy:", chart_accuracy)
format_string_pred = 'llava_fine_tuned/images_new_square/test_{}.py.png'
average_mse = batch_process_images(df, format_string_pred, 'images_new_square/')
print("Average MSE:", average_mse)
# greyscale_average_mse = batch_process_images_bw(df, format_string_pred, 'images_new_square/')
# print("Black & White Average MSE:", greyscale_average_mse)
pred_values = df['values']
average_padding_l1, failed_files = get_l1(pred_values, 'llava_fine_tuned/new_values/test_{}.txt')
print("HistDist:", average_padding_l1)
print("Failed percentage:", len(failed_files)/len(pred_values))
print("Failed files:", failed_files)
pred_categories = df['labels']
average_jaccard, failed_files = get_average_jaccard(pred_categories, 'llava_fine_tuned/new_categories/test_{}.txt')
print("Average Jaccard Similarity:", average_jaccard)
print("Failed percentage:", len(failed_files)/len(pred_categories))
print("Failed files:", failed_files)