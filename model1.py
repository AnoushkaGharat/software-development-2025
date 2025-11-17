# Import libraries and model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "logasanjeev/emotions-analyzer-bert"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Extract label mapping from config
id2label = model.config.id2label  # Example: {0: "joy", 1: "anger", ...}

def predict_emotions(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    top_id = torch.argmax(probs).item()

    return {
        "label": id2label[top_id],
        "confidence": float(probs[top_id])
    }

# Test
text = "I’m so nervous about my exams tomorrow!"
result = predict_emotions(text)

print("Input:", text)
print("Predicted:", result)