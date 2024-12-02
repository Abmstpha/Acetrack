from flask import Flask, request, render_template, redirect, url_for
import os
import pickle
import torch
import PyPDF2
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gdown

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the model path and Google Drive file ID
model_path = "AceTrack_T5.pkl"
gdrive_file_id = "1XYZ1234ABCDEFG"  # Replace with your actual Google Drive file ID
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

# Route: Landing page
@app.route('/')
def index():
    return render_template('index.html')

# Route: Upload and Process PDF
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Parse the PDF
    data = parse_pdf(file_path)

    # Generate feedback
    recommendations = []
    for _, row in data.iterrows():
        feedback = generate_feedback(row['question'], row['correct_answer'], row['student_answer'])
        recommendations.append({'question': row['question'], 'feedback': feedback})

    return render_template('result.html', results=recommendations)

# Parse PDF function
def parse_pdf(file_path):
    data = {'question': [], 'correct_answer': [], 'student_answer': []}
    with open(file_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
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

# Generate feedback function
def generate_feedback(question, correct_answer, student_answer):
    input_text = f"Question: '{question}' Correct Answer: '{correct_answer}' Student Answer: '{student_answer}'"
    inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=100, num_beams=3, early_stopping=True)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    app.run(debug=True)
