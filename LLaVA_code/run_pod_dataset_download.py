import os
import json
from tqdm import tqdm
from datasets import load_dataset

image_dir = "./images"
os.makedirs(image_dir, exist_ok=True)

# Load the dataset and select the first 10 samples
dataset = load_dataset("abhinavl/figure2code_new_data_square", split="train")

# Initialize the list for JSON content
json_content = []

# Iterate over the dataset with a progress bar
for data in tqdm(dataset, desc="Processing images"):
    image = data['image']
    og_file_name = data['og_file_name']
    code = data['code']
    
    # Save the image
    image_path = os.path.join(image_dir, og_file_name)
    image.save(image_path)

    # Create the JSON entry
    json_entry = {
        "id": og_file_name,
        "image": f"{og_file_name}",
        "conversations": [
            {
                "from": "human",
                "value": "Convert the figure you are given into a full code. Here is an example of the expected output:\nimport matplotlib.pyplot as plt\ncategories = ['essay', 'soil']\nvalues = [1, 2]\nplt.figure(figsize=(8, 5))\nplt.bar(categories, values, color='skyblue')\nplt.title('Title')\nplt.xlabel('Categories')\nplt.ylabel('Values')\nplt.show()"
            },
            {
                "from": "gpt",
                "value": code
            }
        ]
    }
    
    # Add the entry to the list
    json_content.append(json_entry)

# Save the JSON file
json_path = "./dataset.json"
os.makedirs(os.path.dirname(json_path), exist_ok=True)
with open(json_path, 'w') as json_file:
    json.dump(json_content, json_file, indent=4)

print(f"Images saved in {image_dir} and JSON file saved as {json_path}")
