import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
model = AutoModelForSequenceClassification.from_pretrained("./saved_model")
model.to(device)

model_data = torch.load("./saved_model/model.pth")
label_encoder_classes = model_data['label_encoder_classes']


def suggest_diseases(user_input):
    inputs = tokenizer(
        user_input, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).flatten()

    top5_probs, top5_indices = torch.topk(probs, 5)
    top5_diseases = np.array(label_encoder_classes)[top5_indices.cpu().numpy()]

    return list(zip(top5_diseases, top5_probs.cpu().numpy()))

st.title("Disease Suggestion Based on Symptoms")
st.write("Enter your symptoms (e.g., fever, headache, sore throat) to get the most likely diseases.")

user_input = st.text_input("Symptoms:")

if user_input:
    st.write("**Processing your input...**")
    suggestions = suggest_diseases(user_input)
    st.write("### Suggested Diseases and Probabilities:")
    for disease, prob in suggestions:
        st.write(f"- **{disease}**: {prob:.4f}")
