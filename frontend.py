import streamlit as st
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import PyPDF2
import os
import gdown

# Define the model path and Google Drive file ID
model_path = "AceTrack_T5.pkl"
gdrive_file_id = "1qw0Zz0CvUP-iEAKNwZ9Qq_LoMlUpijNH"  # Replace with your actual Google Drive file ID
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download the model if not present locally
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={gdrive_file_id}", model_path, quiet=False)

# Load the model and tokenizer
with open(model_path, 'rb') as f:
    saved_objects = torch.load(f, map_location=device)

model = saved_objects['model'].to(device)
tokenizer = saved_objects['tokenizer']

# Function to parse PDF
def parse_pdf(file):
    data = {'question': [], 'correct_answer': [], 'student_answer': []}
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text = page.extract_text().split('\n')
        for line in text:
            if line.startswith("Question:"):
                data['question'].append(line.replace("Question:", "").strip())
            elif line.startswith("Correct Answer:"):
                data['correct_answer'].append(line.replace("Correct Answer:", "").strip())
            elif line.startswith("Student Answer:"):
                data['student_answer'].append(line.replace("Student Answer:", "").strip())
    return pd.DataFrame(data)

# Function to generate feedback
def generate_feedback(question, correct_answer, student_answer):
    input_text = f"Question: '{question}' Correct Answer: '{correct_answer}' Student Answer: '{student_answer}'"
    inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=100, num_beams=3, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.title("AceTrack Feedback Generator")
st.write("Upload a PDF document containing questions, correct answers, and student answers.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    st.write("Processing the uploaded file...")
    parsed_data = parse_pdf(uploaded_file)

    # Display parsed data
    st.write("Extracted Questions, Answers, and Student Responses:")
    st.dataframe(parsed_data)

    # Generate Feedback
    st.write("Generating feedback...")
    feedback_list = []
    for _, row in parsed_data.iterrows():
        feedback = generate_feedback(row['question'], row['correct_answer'], row['student_answer'])
        feedback_list.append({'Question': row['question'], 'Feedback': feedback})

    # Display feedback
    feedback_df = pd.DataFrame(feedback_list)
    st.write("Generated Feedback:")
    st.dataframe(feedback_df)
