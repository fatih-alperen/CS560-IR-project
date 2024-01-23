from transformers import BartForConditionalGeneration, BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments

import json
from torch.utils.data import DataLoader, TensorDataset

# Load pre-trained model and tokenizer
model = BartForConditionalGeneration.from_pretrained("arizona_sky_model")
tokenizer = BartTokenizer.from_pretrained("arizona_sky_model")

# Load your new dataset
def read_json(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
            i = 2
            for line in file:
                i=i+1
                if i%10 == 0:
                    json_obj = json.loads(line)
                    data.append(json_obj)
    return data

# Load test and validation datasets
train_data = read_json('BioLaySumm/task1_development/train/eLife_train.jsonl')
print("loaded train")
# Extract lay_summaries and main_texts from the datasets
train_main_texts = [item['article'] for item in train_data]
train_lay_summaries = [item['lay_summary'] for item in train_data]

print("extracted")


# Tokenize and truncate input texts
train_inputs = tokenizer(train_main_texts, return_tensors="pt", truncation=True, max_length=512, padding=True)

print("tokenized input")

# Tokenize and truncate output summaries
train_labels = tokenizer(train_lay_summaries, return_tensors="pt", truncation=True, max_length=150, padding=True)
print("tokenized labels")

# Create TensorDatasets
train_dataset = TensorDataset(train_inputs["input_ids"], train_inputs["attention_mask"], train_labels["input_ids"], train_labels["attention_mask"])

print("datasets created")


# Define training arguments
training_args = TrainingArguments(
    output_dir="./my_finetuned_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./my_finetuned_model")
tokenizer.save_pretrained("./my_finetuned_model")
