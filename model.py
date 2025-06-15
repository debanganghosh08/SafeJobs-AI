# model.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import os

def load_model_and_tokenizer():
    """Load model and tokenizer with proper error handling"""
    try:
        # Load tokenizer
        if os.path.exists("./bert_tokenizer"):
            tokenizer = BertTokenizer.from_pretrained("./bert_tokenizer")
        else:
            print("Warning: Local tokenizer not found, using default bert-base-uncased")
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # Method 1: Try loading the complete model (recommended)
        if os.path.exists("./bert_model"):
            print("Loading complete model from ./bert_model")
            model = BertForSequenceClassification.from_pretrained("./bert_model")
        else:
            # Method 2: Load state dict with proper architecture matching
            print("Loading state dict from bert_structured_model.pt")
            
            # Check if the .pt file exists
            if not os.path.exists("bert_structured_model.pt"):
                raise FileNotFoundError("bert_structured_model.pt not found")
            
            # Load the state dict first to inspect it
            state_dict = torch.load("bert_structured_model.pt", map_location=torch.device("cpu"))
            
            # Check if it's a complete model or just state dict
            if 'model_state_dict' in state_dict:
                # If saved with additional info
                actual_state_dict = state_dict['model_state_dict']
            elif isinstance(state_dict, dict) and 'bert.embeddings.word_embeddings.weight' in state_dict:
                # If it's a direct state dict
                actual_state_dict = state_dict
            else:
                raise ValueError("Unexpected state dict format")
            
            # Initialize model with correct architecture
            # Check the classifier layer size from state dict
            classifier_weight = actual_state_dict.get('classifier.weight')
            if classifier_weight is not None:
                num_labels = classifier_weight.shape[0]
                print(f"Detected num_labels: {num_labels}")
            else:
                num_labels = 2  # Default for binary classification
                print("Using default num_labels: 2")
            
            # Initialize model
            model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", 
                num_labels=num_labels
            )
            
            # Load state dict
            model.load_state_dict(actual_state_dict, strict=False)
        
        model.eval()
        print("Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Falling back to base model...")
        
        # Fallback: use base model (won't work for predictions but prevents crash)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Function to combine structured fields and tokenize
def preprocess_input(example):
    # Combine structured fields into a single string
    structured_text = (
        f"Title: {example['title']} "
        f"Company Profile: {example['company_profile']} "
        f"Description: {example['description']} "
        f"Requirements: {example['requirements']} "
        f"Location: {example['location']} "
        f"Employment Type: {example['employment_type']} "
        f"Experience: {example['required_experience']} "
        f"Education: {example['required_education']} "
        f"Industry: {example['industry']} "
        f"Function: {example['function']} "
        f"Telecommuting: {example['telecommuting']} "
        f"Company Logo: {example['has_company_logo']} "
        f"Has Questions: {example['has_questions']} "
        f"Salary Range: {example['salary_range']}"
    )
    return structured_text

def predict_single(example):
    try:
        text = preprocess_input(example)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        confidence = probabilities[0][prediction].item()
        
        return {
            "input": example["description"],
            "prediction": prediction,
            "label": "Fake" if prediction == 1 else "Legit",
            "confidence": round(confidence, 4),
            "probabilities": {
                "Legit": round(probabilities[0][0].item(), 4),
                "Fake": round(probabilities[0][1].item(), 4)
            }
        }
    except Exception as e:
        return {
            "input": example["description"],
            "prediction": 0,
            "label": "Error",
            "confidence": 0.0,
            "error": str(e)
        }

def predict_from_dataframe(df):
    results = []
    for _, row in df.iterrows():
        example = {
            "title": str(row.get("title", "")),
            "company_profile": str(row.get("company_profile", "")),
            "description": str(row.get("description", "")),
            "requirements": str(row.get("requirements", "")),
            "location": str(row.get("location", "Unknown")),
            "employment_type": str(row.get("employment_type", "Unknown")),
            "required_experience": str(row.get("required_experience", "Unknown")),
            "required_education": str(row.get("required_education", "Unknown")),
            "industry": str(row.get("industry", "Unknown")),
            "function": str(row.get("function", "Unknown")),
            "telecommuting": int(row.get("telecommuting", 0)),
            "has_company_logo": int(row.get("has_company_logo", 1)),
            "has_questions": int(row.get("has_questions", 0)),
            "salary_range": str(row.get("salary_range", "0-0"))
        }
        result = predict_single(example)
        results.append(result)
    return results

# Optional: wrapper for app use
def predict_from_text(text):
    dummy_input = {
        "title": "",
        "company_profile": "",
        "description": text,
        "requirements": "",
        "location": "Unknown",
        "employment_type": "Unknown",
        "required_experience": "Unknown",
        "required_education": "Unknown",
        "industry": "Unknown",
        "function": "Unknown",
        "telecommuting": 0,
        "has_company_logo": 1,
        "has_questions": 0,
        "salary_range": "0-0"
    }
    return predict_single(dummy_input)

# Debug function to inspect model
def debug_model_info():
    print(f"Model type: {type(model)}")
    print(f"Model config: {model.config}")
    print(f"Number of labels: {model.config.num_labels}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Print model architecture
    print("\nModel architecture:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

if __name__ == "__main__":
    debug_model_info()