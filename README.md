# 🛡️ SafeJobs.AI - Fake Job Scam Detector

![image](https://github.com/user-attachments/assets/bf5064a1-f848-4fc8-aecc-f00dbf1a8f6d)

---

## 📌 Project Overview

**SafeJobs.AI** is an intelligent web app that identifies **fake job listings** using state-of-the-art NLP. Powered by a fine-tuned BERT model, the app takes in a job description and classifies it as either:

- ✅ **Legit**
- ❌ **Fake**

This tool aims to help job seekers avoid scams and identify trustworthy opportunities.

---

## 🚀 Key Features & Technologies Used

### 🔑 Key Features
- 📝 Manual job description analysis
- 📄 CSV file upload for batch processing
- 💡 Clear prediction labels: Legit / Fake
- 📥 Downloadable results (CSV & JSON)
- 📊 Performance metrics and clean interface

### 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Backend:** PyTorch, Transformers (Hugging Face)
- **Model:** `bert-base-uncased` fine-tuned for binary classification
- **Utilities:** Pandas, JSON, CSV handling
- **Deployment:** Streamlit Cloud

---

## 🧠 Model Details

### ⚙️ Architecture
- Model: `BertForSequenceClassification`
- Tokenizer: `BertTokenizer` (`bert-base-uncased`)
- Classes: 2 (0 = Legit, 1 = Fake)
- Max Sequence Length: 512 tokens
- Training Epochs: 4
- Optimizer: AdamW

### 📈 Performance Metrics

| Metric       | Value     |
|--------------|-----------|
| Accuracy     | **97.65%** |
| Precision    | **96.43%** |
| Recall       | **98.12%** |
| F1-Score     | **97.27%** |
| ROC AUC      | **98.88%** |

Trained and evaluated on a cleaned, structured dataset with job descriptions and binary scam labels.

---

## 🌐 Hosted App

🖥️ **Live Demo:**  
👉 [https://safejobs-ai.streamlit.app/](https://safejobs-ai.streamlit.app/)

---

## 📂 GitHub Repository Link

> 📌 Public GitHub Repository includes:
- 🔹 Project code and model scripts
- 🔹 Notebooks (if any preprocessing or training was done in one)
- 🔹 Trained model weights and tokenizer
- 🔹 README with all necessary documentation ✅

🔗 GitHub: [https://github.com/YOUR_USERNAME/safejobs-ai](https://github.com/YOUR_USERNAME/safejobs-ai)

---

## 🛠️ Setup Instructions (Step-by-Step)

> Works on all platforms with Python 3.8+

### 📁 1. Clone the Repository

git clone https://github.com/YOUR_USERNAME/safejobs-ai.git
cd safejobs-ai
📦 2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
📥 3. Ensure Model Files Are Present
Make sure these files/folders are in the root directory:

bert_tokenizer/ → Directory containing tokenizer files

bert_structured_model.pt → Fine-tuned model checkpoint

✅ You can download these from the training pipeline or contact the author for access.

▶️ 4. Run the App Locally
bash
Copy
Edit
streamlit run app.py
Navigate to http://localhost:8501/ to use the app.

📁 Project Structure
bash
Copy
Edit
safejobs-ai/
├── app.py                  # Streamlit frontend
├── model.py                # Model loading and prediction logic
├── utils.py                # CSV & JSON export utils
├── bert_tokenizer/         # Tokenizer directory
├── bert_structured_model.pt # Trained model weights
├── requirements.txt
└── README.md
🧪 Sample Use Cases
✍️ Manual Input
Paste job descriptions like:

text
Copy
Edit
Work-from-home data entry job, ₹5000/day, no interview, sign up now!
📄 CSV Input
Upload a .csv with a description column:

csv
Copy
Edit
description
"Looking for software engineers with 5 years experience in Java"
"Earn ₹10,000/week doing simple typing work from your phone"
💡 Future Work
🌐 Browser Extension for job sites (LinkedIn, Naukri)

🧠 Add probabilistic confidence scores

🗣️ Multilingual support

📊 Admin dashboard for usage analytics

🤝 Feedback system to retrain the model

👨‍💻 Author
Debangan Ghosh
AI & Software Enthusiast
🔗 LinkedIn - https://www.linkedin.com/in/debangan-ghosh/
📧 Email - primusvlog@gmail.com

📜 License
Licensed under the MIT License.

