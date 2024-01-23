from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader, TensorDataset
import torch

import json

# Load tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# model = BartForConditionalGeneration.from_pretrained("arizona_sky_elife/arizona_sky_model_0.1")
# tokenizer = BartTokenizer.from_pretrained("arizona_sky_elife/arizona_sky_model_0.1")


print("loaded tokenizer")

# Read data from JSON files
def read_json(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
            i = 0
            for line in file:
                if i%2 == 0:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                i=i+1
    return data

# Load test and validation datasets

val_data = read_json('BioLaySumm/task1_development/val/PLOS_val.jsonl')

print("loaded validation")

val_main_texts = [item['article'] for item in val_data]
val_lay_summaries = [item['lay_summary'] for item in val_data]


print("extracted")


# Tokenize and truncate input texts
val_inputs = tokenizer(val_main_texts, return_tensors="pt", truncation=True, max_length=512, padding=True)

print("tokenized input")

# Tokenize and truncate output summaries
val_labels = tokenizer(val_lay_summaries, return_tensors="pt", truncation=True, max_length=150, padding=True)

print("tokenized labels")

# Create TensorDatasets
val_dataset = TensorDataset(val_inputs["input_ids"], val_inputs["attention_mask"], val_labels["input_ids"], val_labels["attention_mask"])

print("datasets created")


# DataLoader for batching
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)



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
print(f"Validation Loss: {avg_val_loss}")
with open("crosslog.txt", 'a') as logfile:
    logfile.write(f"Validation Loss: {avg_val_loss}\n")
