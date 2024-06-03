from datasets import load_dataset
# from IPython.display import display

dataset = load_dataset(
    "imagefolder", 
    data_dir="images_square/", 
    split={'train': 'train[:80%]', 'test': 'train[80%:90%]', 'validation': 'train[90%:]'}
)
example = dataset['train'][0]
image = example["image"]
width, height = image.size
print(width, height)
# display(image.resize((int(width), int(height))))
# print(dataset)
dataset.push_to_hub("abhinavl/figure2code_data_square")
