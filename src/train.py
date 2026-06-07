# ============================================================
# train.py — Model Training Script for AI Resume Analyzer
# ============================================================
# This script reads resume data from a CSV file, cleans the
# text, converts it into numerical features using TF-IDF,
# trains a Logistic Regression classifier, evaluates accuracy,
# and saves the trained model + vectorizer to disk so the
# Flask app (app.py) can load and use them for predictions.
# ============================================================

import os                                                  # os module — used for file path operations (joining paths, getting directory names)
import re                                                  # re module — used for regular expressions to clean resume text
import joblib                                              # joblib — used to save (dump) and load trained model files (.pkl)
import pandas as pd                                        # pandas — used to read CSV files and manipulate data in DataFrames
from sklearn.feature_extraction.text import TfidfVectorizer  # TfidfVectorizer — converts raw text into TF-IDF numerical feature vectors
from sklearn.linear_model import LogisticRegression        # LogisticRegression — the ML classification algorithm we use to predict resume categories
from sklearn.metrics import accuracy_score                 # accuracy_score — compares predictions vs actual labels to calculate model accuracy
from sklearn.model_selection import train_test_split       # train_test_split — splits the dataset into training (80%) and testing (20%) portions


def clean_text(text):
    """
    Cleans raw resume text by converting to lowercase,
    removing all special characters/punctuation, and
    collapsing multiple spaces into a single space.
    """
    text = str(text).lower()               # Convert the input to a string (in case it's NaN/number) and make all characters lowercase
    text = re.sub(r'\W', ' ', text)        # Replace every non-word character (punctuation, symbols, etc.) with a space
    text = re.sub(r'\s+', ' ', text)       # Replace multiple consecutive spaces/tabs/newlines with a single space
    return text.strip()                    # Remove any leading/trailing whitespace and return the cleaned text
# ── NOTE: clean_text() ──────────────────────────────────────────────────────────
# This function is the text preprocessing step. It takes raw resume text that may
# contain HTML tags, special characters, inconsistent spacing, and mixed casing,
# and normalizes it into a clean, lowercase, space-separated string of words.
# This is critical because the TF-IDF vectorizer works best with consistent,
# clean text — garbage in = garbage out. The same function is also used in app.py
# so that incoming user resumes are cleaned the exact same way before prediction.
# ────────────────────────────────────────────────────────────────────────────────


def main():
    """
    Main training pipeline: loads data → cleans text → vectorizes →
    trains model → evaluates → saves model files to disk.
    """
    base_dir = os.path.dirname(__file__)                           # Get the directory where this script lives (i.e., the "src/" folder)
    data_path = os.path.join(base_dir, '..', 'data', 'Resume.csv')  # Build the full path to the CSV dataset: src/../data/Resume.csv
    models_dir = os.path.join(base_dir, '..', 'models')           # Build the full path to the models output folder: src/../models/

    # ── STEP 1: Load the dataset ──
    print("Loading dataset...")                                    # Print progress message to the console
    df = pd.read_csv(data_path)                                    # Read the CSV file into a pandas DataFrame (columns: Resume_str, Category, etc.)

    # ── STEP 2: Clean all resume texts ──
    print("Cleaning text...")                                      # Print progress message
    df['cleaned'] = df['Resume_str'].apply(clean_text)             # Apply clean_text() to every row in the 'Resume_str' column and store result in new 'cleaned' column

    # ── STEP 3: Convert text to numerical features using TF-IDF ──
    print("Vectorizing...")                                        # Print progress message
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')  # Create a TF-IDF vectorizer that keeps the top 5000 most important words, ignoring common English stop words (the, is, and, etc.)
    X_tfidf = tfidf.fit_transform(df['cleaned'])                   # Fit the vectorizer on all cleaned resumes and transform them into a sparse matrix of TF-IDF features (each resume becomes a row of 5000 numbers)
    y = df['Category']                                             # Extract the target labels (resume categories like "ENGINEERING", "DESIGNER", etc.)

    # ── STEP 4: Split into training and testing sets ──
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    # Split the data: 80% for training the model, 20% for testing its accuracy
    # random_state=42 ensures the split is the same every time we run the script (reproducibility)

    # ── STEP 5: Train the Logistic Regression model ──
    print("Training model...")                                     # Print progress message
    model = LogisticRegression(max_iter=1000)                      # Create a Logistic Regression classifier with up to 1000 iterations to ensure convergence
    model.fit(X_train, y_train)                                    # Train the model on the training data — it learns which TF-IDF features correspond to which resume categories

    # ── STEP 6: Evaluate the model on test data ──
    y_pred = model.predict(X_test)                                 # Use the trained model to predict categories for the test resumes (ones it hasn't seen before)
    acc = accuracy_score(y_test, y_pred)                           # Compare predictions (y_pred) with actual labels (y_test) to calculate accuracy percentage
    print(f"Accuracy on test set: {acc:.4f}")                      # Print the accuracy rounded to 4 decimal places (e.g., 0.7523 means 75.23% accurate)

    # ── STEP 7: Save trained model and vectorizer to disk ──
    os.makedirs(models_dir, exist_ok=True)                         # Create the "models/" directory if it doesn't already exist (exist_ok=True prevents errors if it exists)
    print("Saving files...")                                       # Print progress message
    joblib.dump(model, os.path.join(models_dir, 'model.pkl'))      # Save the trained Logistic Regression model as "model.pkl" using joblib serialization
    joblib.dump(tfidf, os.path.join(models_dir, 'tfidf.pkl'))      # Save the fitted TF-IDF vectorizer as "tfidf.pkl" (needed to transform new resumes the same way)

    joblib.dump(sorted(df['Category'].unique().tolist()), os.path.join(models_dir, 'categories.pkl'))
    # Save a sorted list of all unique category names (e.g., ["ACCOUNTANT", "DESIGNER", "ENGINEERING", ...]) as "categories.pkl" for reference
    print("Saved successfully!")                                   # Print final success message
# ── NOTE: main() ────────────────────────────────────────────────────────────────
# This is the complete ML training pipeline. It follows the standard workflow:
#   1. Load raw data from CSV
#   2. Preprocess/clean the text
#   3. Convert text → numbers using TF-IDF (Term Frequency-Inverse Document Frequency)
#   4. Split data into train/test sets (80/20 ratio)
#   5. Train a Logistic Regression classifier
#   6. Evaluate accuracy on unseen test data
#   7. Save 3 files to the models/ folder:
#      - model.pkl   → the trained classifier (used for predictions)
#      - tfidf.pkl   → the fitted vectorizer (used to transform new text the same way)
#      - categories.pkl → list of all category names (for reference)
#
# After running this script, app.py loads model.pkl and tfidf.pkl at startup
# to make real-time predictions on user-uploaded resumes.
# ────────────────────────────────────────────────────────────────────────────────


if __name__ == '__main__':   # This block only runs when you execute "python train.py" directly (not when imported as a module)
    main()                   # Call the main() function to start the training pipeline
