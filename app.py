# app.py
import streamlit as st
import pandas as pd
from utils import convert_to_csv, convert_to_json
from model import predict_from_text, predict_from_dataframe

st.set_page_config(page_title="SafeJobs.AI - Job Scam Detector")

st.title("🛡️ SafeJobs.AI")
st.subheader("Detect Fake Job Postings using BERT AI")

option = st.radio("Choose input method:", ["📝 Manual Text", "📄 CSV File"])

if option == "📝 Manual Text":
    job_text = st.text_area("Paste the job description")

    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing..."):
            result = predict_from_text(job_text)
        st.success(f"Prediction: {result['label']}")
        st.json(result)
        st.download_button("📥 Download JSON", convert_to_json(result), file_name="result.json")
        st.download_button("📥 Download CSV", convert_to_csv([result]), file_name="result.csv")

else:
    uploaded_csv = st.file_uploader("Upload a CSV file with a 'description' column", type=["csv"])
    if uploaded_csv and st.button("🔍 Analyze CSV"):
        df = pd.read_csv(uploaded_csv)
        with st.spinner("Analyzing CSV..."):
            results = predict_from_dataframe(df)
        st.success("Analysis complete!")
        st.dataframe(pd.DataFrame(results))
        st.download_button("📥 Download CSV", convert_to_csv(results), file_name="results.csv")
        st.download_button("📥 Download JSON", convert_to_json(results), file_name="results.json")
