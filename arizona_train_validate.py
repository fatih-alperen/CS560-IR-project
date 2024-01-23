from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader, TensorDataset
import torch

import json

# Load tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# model = BartForConditionalGeneration.from_pretrained("arizona_sky_model_0.5")
# tokenizer = BartTokenizer.from_pretrained("arizona_sky_model_0.5")


print("loaded tokenizer")

# Read data from JSON files
def read_json(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
            i = 1
            for line in file:
                if i%5 == 0:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                i=i+1
    return data

# Load test and validation datasets
train_data = read_json('BioLaySumm/task1_development/train/PLOS_train.jsonl')
print("loaded train")
val_data = read_json('BioLaySumm/task1_development/val/PLOS_val.jsonl')

print("loaded validation")

# Extract lay_summaries and main_texts from the datasets
train_main_texts = [item['article'] for item in train_data]
train_lay_summaries = [item['lay_summary'] for item in train_data]

val_main_texts = [item['article'] for item in val_data]
val_lay_summaries = [item['lay_summary'] for item in val_data]


print("extracted")


# Tokenize and truncate input texts
train_inputs = tokenizer(train_main_texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
val_inputs = tokenizer(val_main_texts, return_tensors="pt", truncation=True, max_length=512, padding=True)

print("tokenized input")

# Tokenize and truncate output summaries
train_labels = tokenizer(train_lay_summaries, return_tensors="pt", truncation=True, max_length=150, padding=True)
val_labels = tokenizer(val_lay_summaries, return_tensors="pt", truncation=True, max_length=150, padding=True)

print("tokenized labels")

# Create TensorDatasets
train_dataset = TensorDataset(train_inputs["input_ids"], train_inputs["attention_mask"], train_labels["input_ids"], train_labels["attention_mask"])
val_dataset = TensorDataset(val_inputs["input_ids"], val_inputs["attention_mask"], val_labels["input_ids"], val_labels["attention_mask"])

print("datasets created")


# DataLoader for batching
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Training parameters
epochs = 3
learning_rate = 2e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):

    print(f'training epoch: {epoch}')

    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels_ids, labels_attention_mask = batch

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_ids,
            decoder_attention_mask=labels_attention_mask,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:

            print(f'validating batch: {batch}')
            input_ids, attention_mask, labels_ids, labels_attention_mask = batch

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_ids,
                decoder_attention_mask=labels_attention_mask,
            )

            total_val_loss += outputs.loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}")
    with open("logfile.txt", 'a') as logfile:
        logfile.write(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}\n")

# Save the trained model if needed
model.save_pretrained("arizona_sky_PLOS_0.2")
tokenizer.save_pretrained("arizona_sky_PLOS_0.2")
