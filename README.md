# 🛡️ SafeJobs.AI - Fake Job Scam Detector

![image](https://github.com/user-attachments/assets/bf5064a1-f848-4fc8-aecc-f00dbf1a8f6d)


> 🔍 Detect fraudulent job postings using the power of **BERT AI + Streamlit**  
> 🌐 [Try it Live →](https://safejobs-ai.streamlit.app/)

---

## 📌 Project Overview

**SafeJobs.AI** is an AI-powered application that helps detect **fake job postings** using NLP techniques and a fine-tuned **BERT model**. With a simple UI built on **Streamlit**, users can:

- 📝 Analyze individual job descriptions
- 📄 Upload CSV files with multiple job listings
- 📊 Get instant predictions with downloadable results (CSV/JSON)

---

## 🧠 Model Architecture

This project uses **BERT (Bidirectional Encoder Representations from Transformers)** for sequence classification.  
We fine-tuned `bert-base-uncased` on a binary classification task:  
- `0` → Legitimate job  
- `1` → Scam / Fake job

### 🏗️ Training Process
- ✅ Preprocessing: Lowercasing, truncating/padding to max length 512
- ✅ Tokenization: `BertTokenizer` from HuggingFace
- ✅ Fine-tuning: `BertForSequenceClassification` with 2 output classes
- ✅ Optimizer: AdamW with weight decay
- ✅ Epochs: 4  
- ✅ Device: Trained on GPU (CUDA)

---

## 📈 Model Performance

| Metric       | Value    |
|--------------|----------|
| Accuracy     | **97.65%** |
| Precision    | **96.43%** |
| Recall       | **98.12%** |
| F1-Score     | **97.27%** |
| ROC AUC      | **98.88%** |

> ✅ Evaluation done using test set from the structured dataset with balanced classes.

---

## 🚀 Try it Online

🌍 **Live App:**  
👉 [https://safejobs-ai.streamlit.app/](https://safejobs-ai.streamlit.app/)

---

## 🛠️ Installation Instructions (For Local Setup)

> Prerequisites: Python 3.8+, pip, git, virtualenv (optional but recommended)

### 📁 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/safejobs-ai.git
cd safejobs-ai
