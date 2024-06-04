from datasets import load_dataset
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn



cache_dir = "cache"
dataset = load_dataset('abhinavl/figure2code_data_square', cache_dir = cache_dir)
dataset_train = dataset['train']
dataset_val = dataset['validation']
dataset_test = dataset['test']

from torch.utils.data import Dataset
prompt = (
        "Convert the figure you are given into a full code. Here is an example of the expected output: <N>"
        "import matplotlib.pyplot as plt <N>"
        "# Categories and their corresponding values<N>"
        "categories = ['essay', 'soil']<N>"
        "values = [1, 2]<N>"
        "# Creating the bar chart<N>"
        "plt.figure(figsize=(8, 5))  # Set the figure size (optional)<N>"
        "plt.bar(categories, values, color='skyblue')  # Plot the bars with skyblue color<N>"
        "# Adding title and labels<N>"
        "plt.title('Title')  # Add a title to the chart<N>"
        "plt.xlabel('Categories')  # Label for the x-axis<N>"
        "plt.ylabel('Values')  # Label for the y-axis<N>"
        "# Display the chart<N>"
        "plt.show()"
    )

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        encoding = self.processor(images=item["image"], text=prompt, padding="max_length", return_tensors="pt")
        label = self.processor(text=item["code"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        label = {k: v.squeeze() for k,v in label.items()}
        return encoding, label, item["code"]


from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("microsoft/git-base-vqav2")

test_dataset = ImageCaptioningDataset(dataset_test, processor)
train_dataset = ImageCaptioningDataset(dataset_train, processor)
val_dataset = ImageCaptioningDataset(dataset_val, processor)

from torch.utils.data import DataLoader
batch_size = 1
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vqav2").to('cuda')

from tqdm import tqdm
from codebleu import calc_codebleu

def get_codebleu_scores(targets, preds):
    total_codebleu = 0
    for index in range(len(targets)):
        codebleu_score = calc_codebleu([targets[index]], [preds[index]], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
        total_codebleu += codebleu_score['codebleu']
    return total_codebleu / len(targets)

# Batch size
batch_size_inf = 1

# Function to process a batch of images
def process_batch(batch_images, prompt, model):
    model.eval()
    with torch.no_grad():
      inputs = processor(images=batch_images, text=[prompt] * len(batch_images), return_tensors="pt", padding=True).to('cuda')
      pixel_values = inputs.pixel_values
      generated_ids = model.generate(input_ids=inputs.input_ids, pixel_values=pixel_values, max_length=500)
      generated_codes = processor.batch_decode(generated_ids, skip_special_tokens=True)
      print(generated_codes)
    return generated_codes

def eval_model(model, dataset_test, prompt):
    # List to store the generated captions
    codes = []
    for i in tqdm(range(0, len(dataset_test), batch_size_inf), desc="Evaluating Model"):
        batch_images = [dataset_test[j]['image'] for j in range(i, min(i + batch_size_inf, len(dataset_test)))]
        batch_codes = process_batch(batch_images, prompt, model)
        codes.extend(batch_codes)
    df = pd.DataFrame(codes, columns=["code"])
    codebleu_score = get_codebleu_scores(dataset_test['code'], df['code'])
    return codebleu_score, df
device = 'cuda'
# Training loop

# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
# train_losses=[]
# val_codebleu_scores=[]
# best_val_codebleu=0
# for epoch in range(10):
#     model.train()
#     print("Epoch:", epoch)
#     train_codebleu = 0
#     total_loss = 0
#     num_batches = 0

#     progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
#     for idx, (batch, labels, code) in enumerate(progress_bar):
#         model.train()
#         optimizer.zero_grad()
#         input_ids = batch.pop("input_ids").to(device)
#         attention_mask = batch.pop("attention_mask").to(device)
#         pixel_values = batch.pop("pixel_values").to(device)
#         output_ids = labels.pop("input_ids").to(device)

#         outputs = model(input_ids=input_ids,
#                         attention_mask=attention_mask,
#                         pixel_values=pixel_values,
#                         labels=output_ids)

#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
        

#         total_loss += loss.detach().item()
#         num_batches += 1
#         average_loss = total_loss / num_batches


#         progress_bar.set_postfix({"Average Loss": average_loss})

#     train_losses.append(average_loss)
#     # Evaluate on validation set
#     # val_codebleu, val_df = eval_model(model, dataset_val, prompt)
#     # val_codebleu_scores.append(val_codebleu)
#     # print(f"Validation CodeBLEU: {val_codebleu}")

#     # Save the model
#     # if val_codebleu > best_val_codebleu:
#         # best_val_codebleu = val_codebleu
#     torch.save(model.state_dict(), f"models/model_{epoch}.pth")
#     print("model saved.")

import glob
import os

# def calculate_val_loss(model, val_dataloader, index):
#     model.eval()
#     total_loss = 0
#     num_batches = 0
#     criterion = nn.CrossEntropyLoss()

#     with torch.no_grad():
#         progress_bar = tqdm(val_dataloader, desc=f"Calculating Validation Loss {index}", leave=False)
#         for batch, labels, code in progress_bar:
#             input_ids = batch.pop("input_ids").to(device)
#             attention_mask = batch.pop("attention_mask").to(device)
#             pixel_values = batch.pop("pixel_values").to(device)
#             output_ids = labels.pop("input_ids").to(device)

#             outputs = model(input_ids=input_ids,
#                             attention_mask=attention_mask,
#                             pixel_values=pixel_values,
#                             labels=output_ids)
            
#             loss = outputs.loss
#             total_loss += loss.item()
#             num_batches += 1

#             average_loss = total_loss / num_batches 
#             progress_bar.set_postfix({"Average Loss": average_loss})

#     return average_loss


# Get list of all saved models
# model_files = glob.glob("models/model_*.pth")
# model_files.sort()  # Ensure models are in order

# val_losses = []

# for i, model_file in enumerate(model_files):
#     print(model_file)
#     epoch = int(os.path.basename(model_file).split('_')[1].split('.')[0])
#     model.load_state_dict(torch.load(model_file))
#     model.to()
#     val_loss = calculate_val_loss(model, val_dataloader, i)
#     val_losses.append((epoch, val_loss))
#     print(f"Epoch {epoch}: Validation Loss = {val_loss}")

# # Find the best model (with the lowest validation loss)
# best_epoch, best_val_loss = min(val_losses, key=lambda x: x[1])
# print(f"Best Model: Epoch {best_epoch} with Validation Loss = {best_val_loss}")

# # Plot the validation losses
# plt.figure(figsize=(10, 5))
# plt.plot([x[0] for x in val_losses], [x[1] for x in val_losses], label="Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Validation Loss Over Epochs")
# plt.legend()
# plt.savefig("validation_loss_plot.png")
# plt.show()

# Optionally, load the best model for further use
# model.load_state_dict(torch.load(f"models/model_{best_epoch}.pth"))

# # Plotting loss and CodeBLEU scores
# plt.figure(figsize=(12, 5))

# # plt.subplot(1, 2, 1)
# plt.plot(train_losses, label="Train Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss")
# plt.legend()

# # plt.subplot(1, 2, 2)
# # plt.plot(val_codebleu_scores, label="Validation CodeBLEU")
# # plt.xlabel("Epoch")
# # plt.ylabel("CodeBLEU")
# # plt.title("CodeBLEU Scores")
# # plt.legend()

# # Save the figure
# plt.savefig("loss_scores.png")
# # Batch size
# # List to store the generated captions
# # codes = []


#Load best model:
model.load_state_dict(torch.load("models/model_8.pth"))
model.to('cuda')
print(model)
model.eval()
test_codebleu, df = eval_model(model, dataset_test, prompt)
# df = pd.DataFrame(codes, columns=["Generated_Code"])
print("Test CodeBleu: ", test_codebleu)
# Display the DataFrame
print(df)

# Optionally, you can save the DataFrame to a CSV file
df.to_csv("generated_codes.csv", index=False)