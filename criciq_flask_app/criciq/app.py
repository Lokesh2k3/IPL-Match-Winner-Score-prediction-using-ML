"""
CricIQ — IPL Cricket Analytics Platform
Flask backend  |  Score Predictor (MAE 4.5) + Win Predictor (66% accuracy)
"""

import os, pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# STATIC DATA  (sourced directly from notebooks)
# ─────────────────────────────────────────────────────────────────────────────

TEAMS = sorted([
    "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans",
    "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
    "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
])

VENUES = sorted([
    "ACA-VDCA Cricket Stadium", "Arun Jaitley Stadium",
    "Barsapara Cricket Stadium",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
    "Dr DY Patil Sports Academy", "Eden Gardens",
    "Himachal Pradesh Cricket Association Stadium",
    "M Chinnaswamy Stadium", "MA Chidambaram Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium",
    "Maharashtra Cricket Association Stadium",
    "Narendra Modi Stadium", "Punjab Cricket Association Stadium",
    "Rajiv Gandhi International Stadium", "Sawai Mansingh Stadium",
    "Wankhede Stadium",
])

# Venue 1st-innings averages — notebook cell 19 output (recent seasons)
VENUE_AVG = {
    "ACA-VDCA Cricket Stadium": 211.03,
    "Himachal Pradesh Cricket Association Stadium": 203.56,
    "Arun Jaitley Stadium": 201.55,
    "Punjab Cricket Association Stadium": 198.28,
    "Eden Gardens": 197.75,
    "Narendra Modi Stadium": 196.26,
    "M Chinnaswamy Stadium": 191.75,
    "Rajiv Gandhi International Stadium": 190.35,
    "Wankhede Stadium": 188.04,
    "Sawai Mansingh Stadium": 187.93,
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium": 176.01,
    "Barsapara Cricket Stadium": 174.79,
    "Maharaja Yadavindra Singh International Cricket Stadium": 173.37,
    "MA Chidambaram Stadium": 168.58,
    "Maharashtra Cricket Association Stadium": 172.00,
    "Dr DY Patil Sports Academy": 169.17,
}

# Quality batting remaining dict — notebook cell 50
QUALITY_REMAINING = {
    0: 1.00, 1: 0.82, 2: 0.64, 3: 0.47, 4: 0.32,
    5: 0.20, 6: 0.12, 7: 0.07, 8: 0.04, 9: 0.02,
}

# Team batting / bowling strength (target-encoded from notebook)
GLOBAL_AVG = 171.0
BATTING_STRENGTH = {
    "Chennai Super Kings": 171.5, "Delhi Capitals": 169.2,
    "Gujarat Titans": 174.3, "Kolkata Knight Riders": 175.8,
    "Lucknow Super Giants": 172.1, "Mumbai Indians": 176.4,
    "Punjab Kings": 170.9, "Rajasthan Royals": 173.6,
    "Royal Challengers Bengaluru": 178.2, "Sunrisers Hyderabad": 182.7,
}
BOWLING_STRENGTH = {
    "Chennai Super Kings": 168.3, "Delhi Capitals": 172.4,
    "Gujarat Titans": 167.8, "Kolkata Knight Riders": 170.1,
    "Lucknow Super Giants": 171.3, "Mumbai Indians": 169.7,
    "Punjab Kings": 173.5, "Rajasthan Royals": 170.8,
    "Royal Challengers Bengaluru": 174.9, "Sunrisers Hyderabad": 168.1,
}

# Win predictor data (notebook data_preprocessing.ipynb)
TEAM_WIN_PCT = {
    "Gujarat Titans": 0.617, "Chennai Super Kings": 0.568,
    "Mumbai Indians": 0.553, "Lucknow Super Giants": 0.526,
    "Kolkata Knight Riders": 0.521, "Royal Challengers Bengaluru": 0.500,
    "Rajasthan Royals": 0.498, "Sunrisers Hyderabad": 0.487,
    "Punjab Kings": 0.461, "Delhi Capitals": 0.457,
}

# Head-to-head win pct for (team1, team2) combination
H2H = {
    ("Chennai Super Kings",         "Mumbai Indians"):                0.484,
    ("Chennai Super Kings",         "Kolkata Knight Riders"):         0.516,
    ("Chennai Super Kings",         "Royal Challengers Bengaluru"):   0.565,
    ("Chennai Super Kings",         "Rajasthan Royals"):              0.532,
    ("Chennai Super Kings",         "Delhi Capitals"):                0.571,
    ("Chennai Super Kings",         "Sunrisers Hyderabad"):           0.543,
    ("Chennai Super Kings",         "Punjab Kings"):                  0.516,
    ("Chennai Super Kings",         "Gujarat Titans"):                0.455,
    ("Chennai Super Kings",         "Lucknow Super Giants"):          0.533,
    ("Mumbai Indians",              "Kolkata Knight Riders"):         0.576,
    ("Mumbai Indians",              "Royal Challengers Bengaluru"):   0.576,
    ("Mumbai Indians",              "Rajasthan Royals"):              0.536,
    ("Mumbai Indians",              "Delhi Capitals"):                0.571,
    ("Mumbai Indians",              "Sunrisers Hyderabad"):           0.552,
    ("Mumbai Indians",              "Punjab Kings"):                  0.571,
    ("Kolkata Knight Riders",       "Royal Challengers Bengaluru"):   0.571,
    ("Sunrisers Hyderabad",         "Royal Challengers Bengaluru"):   0.517,
    ("Rajasthan Royals",            "Sunrisers Hyderabad"):           0.517,
    ("Gujarat Titans",              "Chennai Super Kings"):           0.545,
    ("Gujarat Titans",              "Mumbai Indians"):                0.556,
    ("Lucknow Super Giants",        "Chennai Super Kings"):           0.467,
}

TEAM_HOME = {
    "Chennai Super Kings":         "MA Chidambaram Stadium",
    "Mumbai Indians":              "Wankhede Stadium",
    "Royal Challengers Bengaluru": "M Chinnaswamy Stadium",
    "Kolkata Knight Riders":       "Eden Gardens",
    "Delhi Capitals":              "Arun Jaitley Stadium",
    "Punjab Kings":                "Punjab Cricket Association Stadium",
    "Rajasthan Royals":            "Sawai Mansingh Stadium",
    "Sunrisers Hyderabad":         "Rajiv Gandhi International Stadium",
    "Lucknow Super Giants":        "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
    "Gujarat Titans":              "Narendra Modi Stadium",
}

VENUE_CHASING_RATE = {
    "MA Chidambaram Stadium": 0.547, "Punjab Cricket Association Stadium": 0.557,
    "Arun Jaitley Stadium": 0.542,   "Wankhede Stadium": 0.548,
    "Eden Gardens": 0.576,           "M Chinnaswamy Stadium": 0.547,
    "Rajiv Gandhi International Stadium": 0.525,
    "Sawai Mansingh Stadium": 0.641, "Narendra Modi Stadium": 0.531,
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium": 0.538,
    "ACA-VDCA Cricket Stadium": 0.519, "Barsapara Cricket Stadium": 0.543,
    "Maharaja Yadavindra Singh International Cricket Stadium": 0.521,
    "Himachal Pradesh Cricket Association Stadium": 0.556,
    "Maharashtra Cricket Association Stadium": 0.529,
    "Dr DY Patil Sports Academy": 0.543,
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _try_load(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️  Error loading {path}: {e}")
    return None

SCORE_MODEL  = _try_load("ipl_score_model.pkl")
SCORE_FEATS  = _try_load("model_columns.pkl")
WIN_PIPELINE = _try_load("win_model.pkl")   # place your .pkl here

print("✅ Score model loaded" if SCORE_MODEL else "⚠️  Score model not found — demo mode")
print("✅ Win model loaded"   if WIN_PIPELINE else "⚠️  Win model not found — demo mode")

# ─────────────────────────────────────────────────────────────────────────────
# SCORE FEATURE ENGINEERING  (exact replica of notebook cell 50)
# ─────────────────────────────────────────────────────────────────────────────
def build_score_features(batting_team, bowling_team, venue,
                          overs, current_score, wickets_lost):
    vc  = VENUE_AVG.get(venue, GLOBAL_AVG)
    bs  = BATTING_STRENGTH.get(batting_team, GLOBAL_AVG)
    bw  = BOWLING_STRENGTH.get(bowling_team, GLOBAL_AVG)
    crr = current_score / overs if overs > 0 else 0

    balls_completed  = overs * 6
    balls_remaining  = (20 - overs) * 6
    wickets_in_hand  = 10 - wickets_lost
    quality_left     = QUALITY_REMAINING.get(int(min(wickets_lost, 9)), 0.02)

    phase            = 0 if overs <= 6 else (1 if overs <= 15 else 2)
    phase_avg_rr     = {0: 8.5, 1: 8.0, 2: 10.5}[phase]
    rr_vs_phase_norm = crr - phase_avg_rr

    crr_wicket_index      = crr * (wickets_in_hand / 10)
    balls_per_wicket      = balls_completed / (wickets_lost + 1)
    projected_score       = (current_score / balls_completed * 120) if balls_completed > 0 else 0
    runs_to_venue_avg     = vc - current_score
    wicket_adj_projection = current_score + (balls_remaining / 6) * crr * quality_left
    wicket_pressure       = wickets_lost / (overs + 0.1)

    return dict(
        batting_team_strength   = bs,
        bowling_team_strength   = bw,
        venue_first_innings_avg = vc,
        overs_completed         = overs,
        current_score           = current_score,
        wickets_lost            = wickets_lost,
        wickets_in_hand         = wickets_in_hand,
        current_run_rate        = crr,
        crr_wicket_index        = crr_wicket_index,
        balls_per_wicket        = balls_per_wicket,
        balls_remaining         = balls_remaining,
        projected_score         = projected_score,
        runs_to_venue_avg       = runs_to_venue_avg,
        game_phase              = phase,
        quality_batting_left    = quality_left,
        wicket_adj_projection   = wicket_adj_projection,
        wicket_pressure         = wicket_pressure,
        rr_vs_phase_norm        = rr_vs_phase_norm,
    )

def demo_score(feats):
    """Mirrors the RF model's weighted blend logic."""
    wa  = feats["wicket_adj_projection"]
    vav = feats["venue_first_innings_avg"]
    bs  = feats["batting_team_strength"]
    bw  = feats["bowling_team_strength"]

    blend = 0.55 * wa + 0.25 * vav + 0.12 * bs - 0.08 * (bw - GLOBAL_AVG)
    if feats["game_phase"] == 2:   blend *= 1.04
    elif feats["game_phase"] == 0: blend *= 0.96
    return max(int(round(blend)), feats["current_score"] + 5)

# ─────────────────────────────────────────────────────────────────────────────
# WIN PREDICTION HEURISTIC  (mirrors LR model coefficients from notebook)
# ─────────────────────────────────────────────────────────────────────────────
def demo_win(team1, team2, venue, toss_winner, toss_decision):
    t1_wp = TEAM_WIN_PCT.get(team1, 0.50)
    t2_wp = TEAM_WIN_PCT.get(team2, 0.50)
    p1    = 50.0 + (t1_wp - t2_wp) * 40

    # H2H (highest coeff feature in notebook)
    h2h = H2H.get((team1, team2))
    if h2h is None:
        h2h = 1.0 - H2H.get((team2, team1), 0.50)
    p1 += (h2h - 0.50) * 25

    # Home advantage
    if   TEAM_HOME.get(team1) == venue: p1 += 4.5
    elif TEAM_HOME.get(team2) == venue: p1 -= 4.5

    # Chasing / batting first advantage
    team1_chasing = (
        (toss_decision == "field" and toss_winner == team1) or
        (toss_decision == "bat"   and toss_winner != team1)
    )
    cr = VENUE_CHASING_RATE.get(venue, 0.55)
    p1 += (cr - 0.50) * 12 if team1_chasing else -(cr - 0.50) * 12

    # Toss bonus
    if toss_winner == team1: p1 += 2
    if toss_decision == "field":
        p1 += 1.5 if toss_winner == team1 else -1.5

    p1 = round(min(max(p1, 28.0), 72.0), 1)
    return p1, round(100.0 - p1, 1)

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           teams=TEAMS,
                           venues=VENUES,
                           venue_avg=VENUE_AVG)


@app.route("/predict/score", methods=["POST"])
def predict_score():
    d = request.get_json()
    batting_team  = d.get("batting_team", "")
    bowling_team  = d.get("bowling_team", "")
    venue         = d.get("venue", "")
    overs         = float(d.get("overs_completed", 10))
    curr_score    = int(d.get("current_score", 0))
    wickets_lost  = int(d.get("wickets_lost", 0))

    if not all([batting_team, bowling_team, venue]):
        return jsonify({"error": "Please fill in all fields."}), 400
    if batting_team == bowling_team:
        return jsonify({"error": "Batting and bowling teams must be different."}), 400
    if curr_score <= 0:
        return jsonify({"error": "Current score must be greater than 0."}), 400

    feats = build_score_features(batting_team, bowling_team, venue,
                                  overs, curr_score, wickets_lost)

    if SCORE_MODEL is not None:
        cols = list(SCORE_FEATS) if SCORE_FEATS else list(feats.keys())
        row  = pd.DataFrame([{c: feats.get(c, 0) for c in cols}])
        pred = int(round(float(SCORE_MODEL.predict(row)[0])))
        mode = "model"
    else:
        pred = demo_score(feats)
        mode = "demo"

    crr            = round(feats["current_run_rate"], 2)
    overs_rem      = round(20 - overs, 1)
    runs_needed    = max(pred - curr_score, 0)
    req_rr         = round(runs_needed / overs_rem, 2) if overs_rem > 0 else 0
    quality_pct    = int(feats["quality_batting_left"] * 100)
    phase_name     = ["Powerplay", "Middle Overs", "Death Overs"][feats["game_phase"]]

    return jsonify({
        "predicted_score": pred,
        "range_low":        pred - 5,
        "range_high":       pred + 5,
        "current_run_rate": crr,
        "overs_remaining":  overs_rem,
        "runs_needed":      runs_needed,
        "required_rr":      req_rr,
        "quality_pct":      quality_pct,
        "phase":            phase_name,
        "mode":             mode,
    })


@app.route("/predict/win", methods=["POST"])
def predict_win():
    d = request.get_json()
    team1         = d.get("team1", "")
    team2         = d.get("team2", "")
    venue         = d.get("venue", "")
    toss_winner   = d.get("toss_winner", "")
    toss_decision = d.get("toss_decision", "bat")

    if not all([team1, team2, venue, toss_winner]):
        return jsonify({"error": "Please fill in all fields."}), 400
    if team1 == team2:
        return jsonify({"error": "Please select two different teams."}), 400

    if WIN_PIPELINE is not None:
        df_in = pd.DataFrame([{"team1": team1, "team2": team2, "venue": venue,
                                "toss_winner": toss_winner, "toss_decision": toss_decision}])
        probas  = WIN_PIPELINE.predict_proba(df_in)[0]
        classes = list(WIN_PIPELINE.classes_)
        idx1 = classes.index(team1) if team1 in classes else None
        p1   = round(float(probas[idx1]) * 100, 1) if idx1 is not None else 50.0
        p2   = round(100 - p1, 1)
        mode = "model"
    else:
        p1, p2 = demo_win(team1, team2, venue, toss_winner, toss_decision)
        mode   = "demo"

    winner     = team1 if p1 >= p2 else team2
    winner_pct = p1    if p1 >= p2 else p2
    conf = "High" if winner_pct >= 65 else ("Medium" if winner_pct >= 56 else "Low")

    return jsonify({
        "team1": team1, "team2": team2,
        "prob_team1": p1, "prob_team2": p2,
        "winner": winner, "winner_pct": winner_pct,
        "confidence": conf, "mode": mode,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)


