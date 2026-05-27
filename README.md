# IPL Match Winner & Score Prediction Using Machine Learning

## Overview
This project is a Machine Learning-based cricket analytics system developed using historical IPL match data. The system performs:

- Pre-match winner prediction
- Live win probability prediction
- First innings score prediction

The project uses feature engineering and machine learning models to analyze match conditions and generate real-time predictions through a web-based interface.

---

## Features
- IPL Match Winner Prediction
- Live Win Probability Estimation
- First Innings Score Prediction
- Real-time Prediction Interface
- Feature Engineering based ML pipeline
- Flask-based Web Application

---

## Tech Stack
- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML
- CSS
- JavaScript
- Jupyter Notebook

---

## Machine Learning Details
- Prediction Accuracy improved from 55% → 61% → ~70%
- MAE reduced from 13 → 8 → 5 → 3-4 runs

### Feature Engineering Techniques
- Team Win Percentage
- Head-to-Head Statistics
- Recent Form
- Venue Performance
- Elo Rating Difference

---

## Folder Structure


app/
criciq_flask_app/
data/
notebooks/
src/



Installation & Setup

## Step 1: Clone Repository

git clone https://github.com/Lokesh2k3/IPL-Match-Winner-Score-prediction-using-ML.git

---

## Step 2: Move Into Project Folder

cd IPL-Match-Winner-Score-prediction-using-ML

---

## Step 3: Create Virtual Environment

python -m venv venv

---

## Step 4: Activate Virtual Environment

### Windows

venv\Scripts\activate

### Linux / Mac

source venv/bin/activate

---

## Step 5: Install Dependencies

pip install -r requirements.txt

---

## Step 6: Run Flask Application

python app.py

---

## Step 7: Open In Browser

http://127.0.0.1:5000
