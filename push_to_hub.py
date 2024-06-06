from datasets import load_dataset, DatasetDict
# from IPython.display import display
# Load the dataset and shuffle it
dataset = load_dataset("imagefolder", data_dir="images_challenge_square/")
shuffled_dataset = dataset['train'].shuffle(seed=42)  # Shuffle the dataset with a seed for reproducibility

# Split the shuffled dataset
split_dataset = shuffled_dataset.train_test_split(test_size=0.1, seed=42)
train_valid_dataset = split_dataset['train'].train_test_split(test_size=0.1, seed=42)

# Combine the splits
# Combine the splits into a DatasetDict
# final_dataset = DatasetDict({
#     'train': train_valid_dataset['train'],
#     'test': split_dataset['test'],
#     'validation': train_valid_dataset['test']
# })
dataset = load_dataset(
    "imagefolder", 
    data_dir="images_challenge_square/", 
    split='train'
)
# example = dataset['train'][0]
# image = example["image"]
# width, height = image.size
# print(width, height)
# display(image.resize((int(width), int(height))))
# print(dataset)
dataset.push_to_hub("abhinavl/figure2code_challenge_data_square")
