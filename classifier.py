import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModelForMaskedLM
from sklearn.preprocessing import LabelEncoder
import numpy as np

# from transformers import AutoTokenizer, AutoModelForMaskedLM

class SymptomDiseaseDataset(Dataset):
    def __init__(self, data):
        data = data.dropna(subset=['symptoms'])  # Drop rows where 'symptoms' is NaN
        data.loc[:, 'symptoms'] = data['symptoms'].astype(str)  # Ensure all symptoms inputs are strings
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(data['diseases'].values)
        self.texts = data['symptoms'].values
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encodings = self.tokenizer(self.texts[idx], padding='max_length', max_length=128, truncation=True, return_tensors='pt')
        return {
            'input_ids': encodings['input_ids'].flatten(),
            'attention_mask': encodings['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

dataset1 = pd.read_csv('dataset1.csv')
dataset2 = pd.read_csv('dataset2.csv')
dataset3 = pd.read_csv('dataset3.csv')
data = pd.concat([dataset2, dataset3], ignore_index=True)

train_data = SymptomDiseaseDataset(data)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    num_labels=len(train_data.label_encoder.classes_)
)

model.to(device)  # Move model to the selected device

training_args = TrainingArguments(
    output_dir='./results', num_train_epochs=50, per_device_train_batch_size=8,
    logging_dir='./logs', logging_steps=10
)
eval_data = SymptomDiseaseDataset(dataset1)  # Assuming `eval_df` is your evaluation DataFrame
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data  
)


trainer.train()

trainer.save_model("./saved_model")
# Save the trained model as a .pth file
torch.save({
    'model_state_dict': model.state_dict(),
    'label_encoder_classes': train_data.label_encoder.classes_  # Save label encoder classes for later use
}, "./saved_model/model.pth")



print("model successfully saved")
# Define function to suggest diseases based on symptoms
# def suggest_diseases(user_input):
#     # Tokenize user input symptoms
#     inputs = train_data.tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
#     inputs = {k: v.to(device) for k, v in inputs.items()}  # Move input tensors to the selected device
    
#     model.eval()  # Set model to evaluation mode
#     with torch.no_grad():
#         outputs = model(**inputs)
#     probs = torch.nn.functional.softmax(outputs.logits, dim=-1).flatten()

#     # Get top 5 predicted labels and probabilities
#     top5_probs, top5_indices = torch.topk(probs, 5)
#     top5_diseases = train_data.label_encoder.inverse_transform(top5_indices.cpu().numpy())

#     # Display the top 5 diseases with their probabilities
#     print("Suggested diseases based on symptoms:")
#     for disease, prob in zip(top5_diseases, top5_probs.cpu().numpy()):
#         print(f"{disease}: {prob:.4f}")

# # Example usage
# user_symptoms = "I have a rash and joint pain"
# suggest_diseases(user_symptoms)
