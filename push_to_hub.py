from datasets import load_dataset

dataset = load_dataset(
    "imagefolder", 
    data_dir="images/", 
    split={'train': 'train[:80%]', 'test': 'train[80%:90%]', 'validation': 'train[90%:]'}
)
print(dataset)
dataset.push_to_hub("abhinavl/figure2code_data")
