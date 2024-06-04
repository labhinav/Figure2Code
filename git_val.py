import glob
import os

def calculate_val_loss(model, val_dataloader):
    model.eval()
    total_loss = 0
    num_batches = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch, labels, code in val_dataloader:
            input_ids = batch.pop("input_ids").to(device)
            attention_mask = batch.pop("attention_mask").to(device)
            pixel_values = batch.pop("pixel_values").to(device)
            output_ids = labels.pop("input_ids").to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=output_ids)
            
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

    average_loss = total_loss / num_batches
    return average_loss

# Get list of all saved models
model_files = glob.glob("models/model_*.pth")
model_files.sort()  # Ensure models are in order

val_losses = []

for model_file in model_files:
    epoch = int(os.path.basename(model_file).split('_')[1].split('.')[0])
    model.load_state_dict(torch.load(model_file))
    model.to()
    val_loss = calculate_val_loss(model, val_dataloader)
    val_losses.append((epoch, val_loss))
    print(f"Epoch {epoch}: Validation Loss = {val_loss}")

# Find the best model (with the lowest validation loss)
best_epoch, best_val_loss = min(val_losses, key=lambda x: x[1])
print(f"Best Model: Epoch {best_epoch} with Validation Loss = {best_val_loss}")

# Plot the validation losses
plt.figure(figsize=(10, 5))
plt.plot([x[0] for x in val_losses], [x[1] for x in val_losses], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss Over Epochs")
plt.legend()
plt.savefig("validation_loss_plot.png")
plt.show()

# Optionally, load the best model for further use
model.load_state_dict(torch.load(f"models/model_{best_epoch}.pth"))