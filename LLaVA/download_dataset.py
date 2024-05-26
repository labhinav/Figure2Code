from datasets import load_dataset

def download_dataset():
    dataset = load_dataset("abhinavl/figure2code_data", split="test")
    dataset.save_to_disk("local_figure2code_data")

if __name__ == "__main__":
    download_dataset()
