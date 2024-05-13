from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="images/", split="train")
dataset.push_to_hub("abhinavl/figure2code_data")
