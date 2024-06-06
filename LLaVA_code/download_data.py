from datasets import load_dataset, Dataset, DatasetDict
import shutil
import os

# def delete_cache():
#     cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
#     if os.path.exists(cache_dir):
#         shutil.rmtree(cache_dir)
#         print(f"Deleted cache at {cache_dir}")
#     else:
#         print("Cache directory does not exist.")

def download_dataset():
    # Clear cache before downloading datasets
    # delete_cache()

    dataset = load_dataset("abhinavl/figure2code_challenge_data_square", split="train[:1000]")
    dataset.save_to_disk("local_figure2code_challenge_data_square")
    dataset = load_dataset("abhinavl/figure2code_new_data_square", split="test[:1000]")
    dataset.save_to_disk("local_figure2code_new_data_square")

    # dataset = load_dataset("abhinavl/figure2code_challenge_data_square", split="train[:5]")
    # dataset.save_to_disk("local_figure2code_challenge_data_square")
    # dataset = load_dataset("abhinavl/figure2code_new_data_square", split="train[:5]")
    # dataset.save_to_disk("local_figure2code_new_data_square")

if __name__ == "__main__":
    download_dataset()
