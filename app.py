# ============================================================
# app.py — Flask Backend for AI Resume Analyzer
# ============================================================
# This is the main backend server. It loads the pre-trained ML
# model and TF-IDF vectorizer (created by src/train.py), serves
# the frontend HTML page, and exposes two API endpoints:
#   1. /api/analyze      → accepts pasted resume text (JSON)
#   2. /api/analyze-file → accepts uploaded resume files (PDF/DOCX/TXT)
# Both endpoints clean the text, run the ML model, and return
# a JSON response with ATS score, category, metrics, etc.
# ============================================================

import os                                                          # os module — for file path operations and directory handling
import re                                                          # re module — for regular expression text cleaning
from io import BytesIO                                             # BytesIO — wraps raw bytes into a file-like object so PDF/DOCX libraries can read from memory
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
# ThreadPoolExecutor — runs file parsing in a separate thread so it can be timed out if it takes too long
# TimeoutError (aliased as FutureTimeoutError) — raised when the thread exceeds the timeout limit

import joblib                                                      # joblib — loads the saved .pkl model files from disk
from flask import Flask, jsonify, request, send_from_directory     # Flask — the web framework; jsonify converts dicts to JSON responses; request accesses incoming data; send_from_directory serves static files
from flask_cors import CORS                                        # CORS — enables Cross-Origin Resource Sharing so the frontend can call the API from any origin

# ── Optional PDF support ──
try:
    from pypdf import PdfReader                                    # pypdf — library to extract text from PDF files
except ImportError:
    PdfReader = None                                               # If pypdf is not installed, set to None (we check this before using it)

# ── Optional DOCX support ──
try:
    from docx import Document                                      # python-docx — library to extract text from Word (.docx) files
except ImportError:
    Document = None                                                # If python-docx is not installed, set to None (we check this before using it)

# ── Flask App Initialization ──
app = Flask(__name__, static_folder='FRONTEND', static_url_path='')  # Create Flask app; static_folder points to FRONTEND/ directory so HTML/CSS/JS files are served from there
CORS(app)                                                          # Enable CORS on all routes — allows the frontend to make API calls without browser blocking them

# ── Path Constants ──
BASE_DIR = os.path.dirname(__file__)                               # Get the directory where app.py lives (project root)
MODELS_DIR = os.path.join(BASE_DIR, 'models')                     # Build path to models/ folder where model.pkl and tfidf.pkl are stored


def clean_text(text):
    """
    Cleans raw resume text by lowercasing, removing special
    characters, and normalizing whitespace.
    """
    text = str(text).lower()               # Convert input to string and make lowercase (handles NaN, numbers, etc.)
    text = re.sub(r'\W', ' ', text)        # Replace all non-word characters (punctuation, symbols) with spaces
    text = re.sub(r'\s+', ' ', text)       # Collapse multiple spaces/tabs/newlines into a single space
    return text.strip()                    # Remove leading/trailing whitespace and return
# ── NOTE: clean_text() ──────────────────────────────────────────────────────────
# This function normalizes resume text before it's fed into the ML model.
# It must apply the EXACT same transformations used during training (in train.py)
# — otherwise the TF-IDF vectorizer would produce different feature vectors and
# the model's predictions would be meaningless. That's why both files have
# identical clean_text() functions.
# ────────────────────────────────────────────────────────────────────────────────


def load_artifacts():
    """
    Loads the trained ML model and TF-IDF vectorizer from the
    models/ directory. Returns (None, None) if files are missing.
    """
    try:
        model = joblib.load(os.path.join(MODELS_DIR, 'model.pkl'))   # Load the trained Logistic Regression model from disk
        tfidf = joblib.load(os.path.join(MODELS_DIR, 'tfidf.pkl'))   # Load the fitted TF-IDF vectorizer from disk
        return model, tfidf                                          # Return both objects as a tuple
    except Exception:                                                # If files don't exist or are corrupted, catch the error
        print("Model files not found. Run src/train.py first.")      # Print a helpful error message to the console
        return None, None                                            # Return None for both so the app can still start (but API will return 500 errors)
# ── NOTE: load_artifacts() ──────────────────────────────────────────────────────
# This function runs once at server startup (line 116). It loads the two .pkl
# files that train.py created. If they're missing (user hasn't trained yet),
# it gracefully returns None instead of crashing the server. The API route
# handlers check for None and return a helpful "Model is not trained yet" error.
# ────────────────────────────────────────────────────────────────────────────────


def build_response(predicted_category, words):
    """
    Constructs the full JSON response dictionary that the frontend
    expects. Includes ATS score, tags, metrics, improvements, and keywords.
    """
    base_score = 60                                                  # Every resume starts with a base ATS score of 60 out of 100
    length_bonus = min(20, len(words) // 20)                         # Add up to 20 bonus points based on resume length (1 point per 20 words)
    category_words = predicted_category.lower().split()              # Split the predicted category name into individual words (e.g., "DATA SCIENCE" → ["data", "science"])
    keyword_bonus = sum(5 for keyword in category_words if keyword in words)  # Add 5 points for each category keyword found in the resume text
    ats_score = min(98, base_score + length_bonus + keyword_bonus)   # Calculate total ATS score, capped at 98 (never a perfect 100)

    return {
        'ats_score': ats_score,                                      # The overall ATS compatibility score (60-98 range)
        'title': f'Predicted Category: {predicted_category}',        # Headline text shown in the results (e.g., "Predicted Category: ENGINEERING")
        'description': f'Best matching role: {predicted_category}.', # Description text shown below the headline
        'tags': [                                                    # Array of tag badges displayed next to the score
            {'text': f'Category: {predicted_category}', 'cls': 'tag-good'},  # Green tag showing the predicted category
            {'text': 'Length Check', 'cls': 'tag-good' if len(words) > 150 else 'tag-warn'},  # Green if resume has 150+ words, yellow warning if too short
            {'text': 'ML Prediction Ready', 'cls': 'tag-good'},      # Green tag confirming ML model was used
        ],
        'metrics': {                                                 # Three sub-metric scores displayed as progress bars
            'keywords': {'value': f'{min(100, ats_score + 5)}%', 'pct': min(100, ats_score + 5)},   # Keyword match percentage (slightly above ATS score)
            'format': {'value': '80%', 'pct': 80},                   # Format score (fixed at 80% since we don't deeply analyze formatting)
            'readability': {'value': f'{min(100, ats_score - 2)}%', 'pct': min(100, ats_score - 2)},  # Readability score (slightly below ATS score)
        },
        'improvements': [                                            # List of actionable suggestions for the user
            {
                'icon': '⚡',                                        # Lightning bolt emoji icon for this suggestion
                'level': 'med',                                      # Priority level CSS class (medium = yellow)
                'title': f'Add more {predicted_category} keywords',  # Suggestion title customized with the predicted category
                'desc': 'Use job-specific words from real job descriptions.',  # Detailed description of the suggestion
                'priority': 'MEDIUM',                                # Priority label text
            },
            {
                'icon': '📊',                                        # Chart emoji icon
                'level': 'low',                                      # Priority level CSS class (low = green)
                'title': 'Use measurable achievements',              # Suggestion title
                'desc': 'Add numbers like percentages, time saved, or targets reached.',  # Detailed description
                'priority': 'LOW',                                   # Priority label text
            },
        ],
        'keywords': [{'word': word.capitalize(), 'found': True} for word in category_words] + [
            # Generate keyword chips: first, mark all category words as "found" (since the model predicted this category)
            {'word': 'Leadership', 'found': 'leadership' in words},      # Check if "leadership" appears in the resume
            {'word': 'Communication', 'found': 'communication' in words},  # Check if "communication" appears in the resume
            {'word': 'Teamwork', 'found': 'team' in words or 'teamwork' in words},  # Check if "team" or "teamwork" appears
        ],
    }
# ── NOTE: build_response() ──────────────────────────────────────────────────────
# This function takes the ML model's raw prediction (a category string) and the
# list of words from the cleaned resume, then constructs the complete JSON object
# that the frontend's populateResults() JavaScript function expects. It calculates
# a simple ATS score based on resume length and keyword presence, generates
# improvement suggestions, and checks for common soft-skill keywords. The response
# structure must match exactly what the frontend expects (ats_score, title,
# description, tags, metrics, improvements, keywords) — changing field names
# here would break the frontend display.
# ────────────────────────────────────────────────────────────────────────────────


def analyze_text(text):
    """
    The core analysis pipeline: clean text → vectorize → predict → build response.
    """
    cleaned_text = clean_text(text)                  # Clean the raw resume text (lowercase, remove special chars, normalize spaces)
    features = tfidf.transform([cleaned_text])       # Transform the cleaned text into TF-IDF feature vector using the pre-fitted vectorizer (same vocabulary as training)
    prediction = model.predict(features)[0]          # Run the feature vector through the trained Logistic Regression model to get the predicted category (e.g., "ENGINEERING")
    return build_response(str(prediction), cleaned_text.split())  # Build and return the full response dict, passing the prediction and the list of individual words
# ── NOTE: analyze_text() ────────────────────────────────────────────────────────
# This is the main analysis function called by both API endpoints. It chains
# together: text cleaning → TF-IDF vectorization → model prediction → response
# building. The tfidf.transform() call uses the SAME vocabulary learned during
# training (saved in tfidf.pkl), so new text is mapped to the same feature space.
# model.predict() returns an array of predictions; [0] gets the first (only) one.
# ────────────────────────────────────────────────────────────────────────────────


def extract_text_from_file(filename, raw_bytes):
    """
    Extracts plain text from an uploaded file based on its extension.
    Supports .txt, .pdf, and .docx formats.
    Returns None for unsupported file types.
    """
    ext = os.path.splitext((filename or '').lower())[1]  # Get the file extension in lowercase (e.g., ".pdf", ".docx", ".txt")

    if ext == '.txt':                                     # If it's a plain text file:
        return raw_bytes.decode('utf-8', errors='ignore')  # Decode the raw bytes to a UTF-8 string (ignore any invalid characters)

    if ext == '.pdf':                                     # If it's a PDF file:
        if PdfReader is None:                             # Check if pypdf library is installed
            raise RuntimeError("PDF support not installed. Run: pip install pypdf")  # Raise error if not installed
        reader = PdfReader(BytesIO(raw_bytes))            # Create a PDF reader from the raw bytes (BytesIO wraps bytes into a file-like object)
        return '\n'.join((page.extract_text() or '') for page in reader.pages)  # Extract text from each page and join with newlines

    if ext == '.docx':                                    # If it's a Word document:
        if Document is None:                              # Check if python-docx library is installed
            raise RuntimeError("DOCX support not installed. Run: pip install python-docx")  # Raise error if not installed
        document = Document(BytesIO(raw_bytes))           # Create a Document object from the raw bytes
        return '\n'.join(paragraph.text for paragraph in document.paragraphs)  # Extract text from each paragraph and join with newlines

    return None                                           # If the file extension is not .txt, .pdf, or .docx, return None (unsupported format)
# ── NOTE: extract_text_from_file() ──────────────────────────────────────────────
# This function handles the file-to-text conversion for all three supported
# formats. It receives the filename and raw bytes (already read from the upload
# stream on the main thread) and parses them in a background thread. This design
# allows us to set a timeout — if a large/corrupt PDF takes too long to parse,
# the thread is cancelled and the user gets a timeout error instead of the
# server hanging forever. The function returns None for unsupported file types,
# which the calling route handler checks and returns an appropriate error.
# ────────────────────────────────────────────────────────────────────────────────


# ── Load the trained model at server startup ──
model, tfidf = load_artifacts()                          # Load model.pkl and tfidf.pkl when the server starts; both are None if files are missing


# ============================================================
#  ROUTE HANDLERS (API Endpoints)
# ============================================================

@app.route('/')                                          # Register the root URL route "/"
def index():
    """Serves the main frontend HTML page."""
    return send_from_directory(app.static_folder, 'ai_resume_analyzer.html')  # Send the HTML file from the FRONTEND/ folder to the browser
# ── NOTE: index() ───────────────────────────────────────────────────────────────
# This route serves the single-page frontend. When a user visits
# http://localhost:5000/, Flask sends the ai_resume_analyzer.html file from the
# FRONTEND/ directory. The static_folder was set to 'FRONTEND' during app
# initialization (line where Flask app is created), so send_from_directory
# knows where to look.
# ────────────────────────────────────────────────────────────────────────────────


@app.route('/api/analyze', methods=['POST'])              # Register the POST endpoint "/api/analyze" for paste-text analysis
def analyze():
    """
    API endpoint for analyzing pasted resume text.
    Expects JSON body: { "text": "resume content here..." }
    Returns JSON with ATS score, category, metrics, etc.
    """
    if model is None or tfidf is None:                    # Check if model files were loaded successfully at startup
        return jsonify({'error': 'Model is not trained yet.'}), 500  # Return 500 error if model is missing

    data = request.get_json(silent=True) or {}            # Parse the JSON body from the request; silent=True returns None instead of raising error if JSON is invalid; fallback to empty dict
    text = data.get('text', '')                           # Extract the 'text' field from the JSON body; default to empty string if not provided
    if not text.strip():                                  # Check if the text is empty or only whitespace
        return jsonify({'error': 'No text provided.'}), 400  # Return 400 bad request error if no text

    return jsonify(analyze_text(text))                    # Run the analysis pipeline and return the result as a JSON response
# ── NOTE: analyze() ─────────────────────────────────────────────────────────────
# This endpoint handles the "Paste Text" tab in the frontend. The JavaScript
# sends a POST request with JSON body containing the resume text. The endpoint
# validates the input, runs it through the ML pipeline (clean → vectorize →
# predict → build response), and returns the full analysis result as JSON.
# The frontend's populateResults() function then renders this data on screen.
# ────────────────────────────────────────────────────────────────────────────────


@app.route('/api/analyze-file', methods=['POST'])         # Register the POST endpoint "/api/analyze-file" for file upload analysis
def analyze_file():
    """
    API endpoint for analyzing uploaded resume files.
    Expects multipart form data with a 'file' field.
    Supports .pdf, .docx, and .txt files.
    Returns JSON with ATS score, category, metrics, etc.
    """
    if model is None or tfidf is None:                    # Check if model files were loaded successfully at startup
        return jsonify({'error': 'Model is not trained yet.'}), 500  # Return 500 error if model is missing

    uploaded_file = request.files.get('file')             # Get the uploaded file from the multipart form data (key name is 'file')
    if uploaded_file is None or not uploaded_file.filename:  # Check if a file was actually uploaded and has a filename
        return jsonify({'error': 'No file uploaded.'}), 400  # Return 400 error if no file

    raw_bytes = uploaded_file.read()                      # Read ALL bytes from the upload stream into memory (must be done on main thread before passing to background thread)
    filename = uploaded_file.filename                     # Save the original filename (needed to detect file extension)

    executor = ThreadPoolExecutor(max_workers=1)          # Create a thread pool with 1 worker thread for file parsing
    try:
        future = executor.submit(extract_text_from_file, filename, raw_bytes)  # Submit the file parsing task to the background thread
        text = future.result(timeout=12)                  # Wait up to 12 seconds for the result; raises TimeoutError if it takes longer
    except RuntimeError as error:                         # Catch RuntimeError (raised when PDF/DOCX library is not installed)
        return jsonify({'error': str(error)}), 500        # Return 500 with the specific error message
    except FutureTimeoutError:                            # Catch timeout — file parsing took longer than 12 seconds
        future.cancel()                                   # Cancel the still-running task
        return jsonify({'error': 'File processing took too long. Try DOCX/TXT or paste text.'}), 408  # Return 408 Request Timeout
    except Exception as e:                                # Catch any other unexpected error during file parsing
        import traceback                                  # Import traceback module to print the full error stack trace
        traceback.print_exc()                             # Print the full error details to the server console (for debugging)
        return jsonify({'error': f'Could not process this file. {str(e)}'}), 400  # Return 400 with the error message
    finally:
        executor.shutdown(wait=False, cancel_futures=True)  # Always clean up the thread pool (don't wait for running tasks, cancel any pending ones)

    if text is None:                                      # Check if extract_text_from_file returned None (unsupported file type)
        return jsonify({'error': 'Only .txt, .pdf, and .docx files are supported.'}), 400  # Return 400 error
    if not text.strip():                                  # Check if the extracted text is empty (e.g., scanned PDF with no selectable text)
        return jsonify({'error': 'Could not extract text from this file.'}), 400  # Return 400 error

    return jsonify(analyze_text(text))                    # Run the analysis pipeline on the extracted text and return JSON response
# ── NOTE: analyze_file() ────────────────────────────────────────────────────────
# This endpoint handles the "Upload File" tab in the frontend. The JavaScript
# sends the file as multipart form data. The key design decisions are:
#   1. raw_bytes are read on the MAIN thread (line with uploaded_file.read()) —
#      this is critical because Flask's file stream is NOT safe to read from a
#      background thread (the request context may be gone).
#   2. File parsing runs in a BACKGROUND THREAD with a 12-second timeout —
#      this prevents large/corrupt files from hanging the server indefinitely.
#   3. Multiple error types are caught separately (RuntimeError for missing
#      libraries, TimeoutError for slow files, generic Exception as fallback)
#      so the user gets specific, helpful error messages.
# ────────────────────────────────────────────────────────────────────────────────


# ── Start the server ──
if __name__ == '__main__':                                # Only runs when executing "python app.py" directly (not when imported)
    app.run(debug=True, port=5000)                        # Start the Flask development server on port 5000 with debug mode ON (auto-reloads on code changes, shows detailed errors)
# ── NOTE: Server Startup ───────────────────────────────────────────────────────
# debug=True enables two helpful features during development:
#   1. Auto-reload: the server restarts automatically when you save code changes
#   2. Debug pages: if an error occurs, Flask shows a detailed error page in the
#      browser instead of a generic "500 Internal Server Error"
# For production deployment, debug should be set to False and a proper WSGI
# server like Gunicorn or Waitress should be used instead of Flask's built-in server.
# ────────────────────────────────────────────────────────────────────────────────
