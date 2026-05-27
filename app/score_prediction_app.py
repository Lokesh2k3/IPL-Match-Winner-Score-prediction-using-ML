import streamlit as st
import pickle
import pandas as pd

# Load model and column structure
model = pickle.load(open("notebooks/score_prediction_model.pkl", "rb"))
model_columns = pickle.load(open("notebooks/model_columns.pkl", "rb"))

st.title("IPL Score Prediction")

teams = [
"Chennai Super Kings",
"Delhi Capitals",
"Gujarat Titans",
"Kolkata Knight Riders",
"Lucknow Super Giants",
"Mumbai Indians",
"Punjab Kings",
"Rajasthan Royals",
"Royal Challengers Bengaluru",
"Sunrisers Hyderabad"
]

venues = [
"M Chinnaswamy Stadium",
"Wankhede Stadium",
"Eden Gardens",
"Arun Jaitley Stadium",
"Narendra Modi Stadium",
"MA Chidambaram Stadium",
"Sawai Mansingh Stadium",
"Rajiv Gandhi International Stadium"
]

batting_team = st.selectbox("Batting Team", teams)
bowling_team = st.selectbox("Bowling Team", teams)
venue = st.selectbox("Venue", venues)

overs_completed = st.number_input(
"Overs Completed",
min_value=0.1,
max_value=20.0,
step=0.1,
format="%.1f"
)

current_score = st.number_input("Current Score", min_value=0)
wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=10)

if st.button("Predict Score"):

    current_run_rate = current_score / overs_completed

    input_df = pd.DataFrame({
        "batting_team":[batting_team],
        "bowling_team":[bowling_team],
        "venue":[venue],
        "overs_completed":[overs_completed],
        "current_score":[current_score],
        "wickets_lost":[wickets_lost],
        "current_run_rate":[current_run_rate],
        "venue_first_innings_avg":[180]
    })

    # Convert categorical → one-hot
    input_encoded = pd.get_dummies(input_df)

    # Ensure same columns as training
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_encoded)

    st.success(f"Predicted Final Score: {int(prediction[0])}")