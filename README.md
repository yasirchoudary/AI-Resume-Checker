<<<<<<< HEAD
# AI Resume Analyzer (Semester Project)

Flask + ML project that predicts resume category and shows an ATS-style report.

## Project Structure

- `app.py` - Flask backend API + frontend hosting
- `src/train.py` - model training script
- `FRONTEND/ai_resume_analyzer.html` - UI (upload + paste + results)
- `data/Resume.csv` - training dataset
- `models/` - saved model files (`model.pkl`, `tfidf.pkl`, `categories.pkl`)

## Setup

1. Create virtual environment (optional but recommended)
2. Install packages:

```bash
pip install -r requirements.txt
```

## Train Model

```bash
python src/train.py
```

This creates trained files inside `models/`.

## Run App

```bash
python app.py
```

Open: [http://localhost:5000](http://localhost:5000)

## Features

- **File Upload**: Upload PDF, DOCX, or TXT resumes for analysis
- **Paste Text**: Paste resume content directly
- **ML Prediction**: Logistic Regression model predicts resume category
- **ATS Score**: Generated score with keyword, format, and readability metrics
- **Export Report**: Download analysis results as a text file
=======
# AI-Resume-Checker
Python based AI-Resume Checker
>>>>>>> 3e2cd9386075e643f630a83909664e45e2558eb3
