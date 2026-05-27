"""
CricIQ — IPL Cricket Analytics Platform
Flask backend  |  Score Predictor + Win Predictor + Win Probability (Live)

CRICKET LOGIC RULES ENFORCED:
  1. Overs are always whole numbers (1–19). No decimals.
  2. Same score, more overs = LOWER prediction (CRR dropped = slower scoring).
  3. Wickets dominate tail predictions — 9 wickets gone → barely 10 more runs.
  4. Real model output is blended with cricket-aware logic to prevent direction errors.
"""

import os, pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# STATIC DATA
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

# Quality batting remaining
QUALITY_REMAINING = {
    0: 1.00, 1: 0.82, 2: 0.64, 3: 0.47, 4: 0.32,
    5: 0.20, 6: 0.12, 7: 0.07, 8: 0.04, 9: 0.02,
}

EXPECTED_RPO = {
    10: 11.5, 9: 11.0, 8: 10.5, 7: 9.5,
     6:  9.0, 5:  8.5, 4:  8.0, 3: 7.0,
     2:  5.0, 1:  3.0, 0:  0.0,
}

ABS_MAX_ADD_14OV = {
     0:   0,  1:  12,  2:  18,  3:  38,
     4:  50,  5:  88,  6:  98,  7: 130,
     8: 147,  9: 154, 10: 161,
}

GLOBAL_AVG = 171.0
BATTING_STRENGTH = {
    "Chennai Super Kings": 171.5,        "Delhi Capitals": 169.2,
    "Gujarat Titans": 174.3,             "Kolkata Knight Riders": 175.8,
    "Lucknow Super Giants": 172.1,       "Mumbai Indians": 176.4,
    "Punjab Kings": 170.9,               "Rajasthan Royals": 173.6,
    "Royal Challengers Bengaluru": 178.2,"Sunrisers Hyderabad": 182.7,
}
BOWLING_STRENGTH = {
    "Chennai Super Kings": 168.3,        "Delhi Capitals": 172.4,
    "Gujarat Titans": 167.8,             "Kolkata Knight Riders": 170.1,
    "Lucknow Super Giants": 171.3,       "Mumbai Indians": 169.7,
    "Punjab Kings": 173.5,               "Rajasthan Royals": 170.8,
    "Royal Challengers Bengaluru": 174.9,"Sunrisers Hyderabad": 168.1,
}

TEAM_WIN_PCT = {
    "Gujarat Titans": 0.617,        "Chennai Super Kings": 0.568,
    "Mumbai Indians": 0.553,        "Lucknow Super Giants": 0.526,
    "Kolkata Knight Riders": 0.521, "Royal Challengers Bengaluru": 0.500,
    "Rajasthan Royals": 0.498,      "Sunrisers Hyderabad": 0.487,
    "Punjab Kings": 0.461,          "Delhi Capitals": 0.457,
}

H2H = {
    ("Chennai Super Kings",        "Mumbai Indians"):               0.484,
    ("Chennai Super Kings",        "Kolkata Knight Riders"):        0.516,
    ("Chennai Super Kings",        "Royal Challengers Bengaluru"):  0.565,
    ("Chennai Super Kings",        "Rajasthan Royals"):             0.532,
    ("Chennai Super Kings",        "Delhi Capitals"):               0.571,
    ("Chennai Super Kings",        "Sunrisers Hyderabad"):          0.543,
    ("Chennai Super Kings",        "Punjab Kings"):                 0.516,
    ("Chennai Super Kings",        "Gujarat Titans"):               0.455,
    ("Chennai Super Kings",        "Lucknow Super Giants"):         0.533,
    ("Mumbai Indians",             "Kolkata Knight Riders"):        0.576,
    ("Mumbai Indians",             "Royal Challengers Bengaluru"):  0.576,
    ("Mumbai Indians",             "Rajasthan Royals"):             0.536,
    ("Mumbai Indians",             "Delhi Capitals"):               0.571,
    ("Mumbai Indians",             "Sunrisers Hyderabad"):          0.552,
    ("Mumbai Indians",             "Punjab Kings"):                 0.571,
    ("Kolkata Knight Riders",      "Royal Challengers Bengaluru"):  0.571,
    ("Sunrisers Hyderabad",        "Royal Challengers Bengaluru"):  0.517,
    ("Rajasthan Royals",           "Sunrisers Hyderabad"):          0.517,
    ("Gujarat Titans",             "Chennai Super Kings"):          0.545,
    ("Gujarat Titans",             "Mumbai Indians"):               0.556,
    ("Lucknow Super Giants",       "Chennai Super Kings"):          0.467,
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
    "MA Chidambaram Stadium": 0.547,
    "Punjab Cricket Association Stadium": 0.557,
    "Arun Jaitley Stadium": 0.542,
    "Wankhede Stadium": 0.548,
    "Eden Gardens": 0.576,
    "M Chinnaswamy Stadium": 0.547,
    "Rajiv Gandhi International Stadium": 0.525,
    "Sawai Mansingh Stadium": 0.641,
    "Narendra Modi Stadium": 0.531,
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium": 0.538,
    "ACA-VDCA Cricket Stadium": 0.519,
    "Barsapara Cricket Stadium": 0.543,
    "Maharaja Yadavindra Singh International Cricket Stadium": 0.521,
    "Himachal Pradesh Cricket Association Stadium": 0.556,
    "Maharashtra Cricket Association Stadium": 0.529,
    "Dr DY Patil Sports Academy": 0.543,
}

# ── Venue 2nd innings averages (from win_probability notebook) ──
VENUE_2ND_AVG = {
    "ACA-VDCA Cricket Stadium": 205.0,
    "Arun Jaitley Stadium": 198.5,
    "Barsapara Cricket Stadium": 170.0,
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium": 172.0,
    "Dr DY Patil Sports Academy": 165.0,
    "Eden Gardens": 193.0,
    "Himachal Pradesh Cricket Association Stadium": 197.0,
    "M Chinnaswamy Stadium": 187.0,
    "MA Chidambaram Stadium": 165.0,
    "Maharaja Yadavindra Singh International Cricket Stadium": 168.0,
    "Maharashtra Cricket Association Stadium": 168.0,
    "Narendra Modi Stadium": 191.0,
    "Punjab Cricket Association Stadium": 194.0,
    "Rajiv Gandhi International Stadium": 185.0,
    "Sawai Mansingh Stadium": 183.0,
    "Wankhede Stadium": 184.0,
}

OVERALL_CHASE_RATE = 0.50   # fallback when no h2h data found

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

SCORE_MODEL   = _try_load("../notebooks/ipl_score_model.pkl")
SCORE_FEATS   = _try_load("../notebooks/model_columns.pkl")
WIN_PIPELINE  = _try_load("../notebooks/win_model.pkl")

# ── Win Probability (2nd innings) model files ──
WIN_PROB_MODEL   = _try_load("../notebooks/win_probability_model.pkl")
WIN_PROB_COLS    = _try_load("../notebooks/win_model_columns.pkl")
H2H_TABLE        = _try_load("../notebooks/h2h_table.pkl")
VENUE_2ND_DF     = _try_load("../notebooks/venue_2nd_avg.pkl")
VENUE_CHASE_DF   = _try_load("../notebooks/venue_chase_rate.pkl")

print("✅ Score model loaded"        if SCORE_MODEL    else "⚠️  Score model not found — demo mode")
print("✅ Win model loaded"          if WIN_PIPELINE   else "⚠️  Win model not found — demo mode")
print("✅ Win Probability model loaded" if WIN_PROB_MODEL else "⚠️  Win Probability model not found — demo mode")

# ─────────────────────────────────────────────────────────────────────────────
# CRICKET-AWARE SCORE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def cricket_aware_score(overs, score, wkts):
    crr       = score / overs if overs > 0 else 0
    overs_rem = 20 - overs
    wih       = 10 - wkts
    ql        = QUALITY_REMAINING.get(int(min(wkts, 9)), 0.02)
    exp_rpo   = EXPECTED_RPO.get(wih, 3.0)

    if wih >= 6:
        crr_proj   = crr * (0.70 + 0.30 * (wih / 10))
        crr_weight = 0.75
    elif wih >= 3:
        crr_proj   = crr * (0.25 + 0.75 * ql)
        crr_weight = 0.60
    else:
        crr_proj   = crr * ql
        crr_weight = 0.30

    future_rpo = crr_weight * crr_proj + (1 - crr_weight) * exp_rpo

    if overs >= 15 and wih >= 5:
        future_rpo = max(future_rpo, exp_rpo * 0.88)
    if overs >= 18 and wih >= 4:
        future_rpo = max(future_rpo, exp_rpo * 0.95)

    raw_proj   = score + overs_rem * future_rpo
    overs_frac = min(overs_rem / 14.0, 1.0)
    max_add    = ABS_MAX_ADD_14OV.get(wih, 10) * overs_frac
    ceiling    = score + max_add

    return int(round(max(min(raw_proj, ceiling), score + 1)))


def build_model_features(batting_team, bowling_team, venue, overs, score, wkts):
    vc  = VENUE_AVG.get(venue, GLOBAL_AVG)
    bs  = BATTING_STRENGTH.get(batting_team, GLOBAL_AVG)
    bw  = BOWLING_STRENGTH.get(bowling_team, GLOBAL_AVG)
    crr = score / overs if overs > 0 else 0

    balls_completed       = overs * 6
    balls_remaining       = (20 - overs) * 6
    wih                   = 10 - wkts
    ql                    = QUALITY_REMAINING.get(int(min(wkts, 9)), 0.02)
    phase                 = 0 if overs <= 6 else (1 if overs <= 15 else 2)
    phase_avg_rr          = {0: 8.5, 1: 8.0, 2: 10.5}[phase]
    rr_vs_phase_norm      = crr - phase_avg_rr
    crr_wicket_index      = crr * (wih / 10)
    balls_per_wicket      = balls_completed / (wkts + 1)
    projected_score       = (score / balls_completed * 120) if balls_completed > 0 else 0
    runs_to_venue_avg     = vc - score
    wicket_adj_projection = score + (balls_remaining / 6) * crr * ql
    wicket_pressure       = wkts / (overs + 0.1)

    return dict(
        batting_team_strength   = bs,
        bowling_team_strength   = bw,
        venue_first_innings_avg = vc,
        overs_completed         = overs,
        current_score           = score,
        wickets_lost            = wkts,
        wickets_in_hand         = wih,
        current_run_rate        = crr,
        crr_wicket_index        = crr_wicket_index,
        balls_per_wicket        = balls_per_wicket,
        balls_remaining         = balls_remaining,
        projected_score         = projected_score,
        runs_to_venue_avg       = runs_to_venue_avg,
        game_phase              = phase,
        quality_batting_left    = ql,
        wicket_adj_projection   = wicket_adj_projection,
        wicket_pressure         = wicket_pressure,
        rr_vs_phase_norm        = rr_vs_phase_norm,
    )


# ─────────────────────────────────────────────────────────────────────────────
# WIN PREDICTION HEURISTIC (Pre-match)
# ─────────────────────────────────────────────────────────────────────────────
def demo_win(team1, team2, venue, toss_winner, toss_decision):
    t1_wp = TEAM_WIN_PCT.get(team1, 0.50)
    t2_wp = TEAM_WIN_PCT.get(team2, 0.50)
    p1    = 50.0 + (t1_wp - t2_wp) * 40

    h2h = H2H.get((team1, team2))
    if h2h is None:
        h2h = 1.0 - H2H.get((team2, team1), 0.50)
    p1 += (h2h - 0.50) * 25

    if   TEAM_HOME.get(team1) == venue: p1 += 4.5
    elif TEAM_HOME.get(team2) == venue: p1 -= 4.5

    team1_chasing = (
        (toss_decision == "field" and toss_winner == team1) or
        (toss_decision == "bat"   and toss_winner != team1)
    )
    cr = VENUE_CHASING_RATE.get(venue, 0.55)
    p1 += (cr - 0.50) * 12 if team1_chasing else -(cr - 0.50) * 12

    if toss_winner == team1: p1 += 2
    if toss_decision == "field":
        p1 += 1.5 if toss_winner == team1 else -1.5

    p1 = round(min(max(p1, 28.0), 72.0), 1)
    return p1, round(100.0 - p1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# WIN PROBABILITY HEURISTIC (Live 2nd innings — demo mode fallback)
# ─────────────────────────────────────────────────────────────────────────────
def demo_win_probability(batting_team, bowling_team, venue,
                         target, current_score, wickets_lost, overs_completed):
    """
    Cricket-logic based win probability when real model is not available.
    Uses RRR vs CRR pressure, wickets remaining, venue chasing rate and H2H.
    """
    runs_required     = max(target - current_score, 0)
    balls_remaining   = max(120 - overs_completed * 6, 1)
    required_run_rate = runs_required / (balls_remaining / 6)
    current_run_rate  = current_score / overs_completed if overs_completed > 0 else 0
    rrr_vs_crr        = required_run_rate - current_run_rate   # +ve = chasing team under pressure
    wickets_remaining = 10 - wickets_lost

    # Base: venue chasing rate
    base = VENUE_CHASING_RATE.get(venue, OVERALL_CHASE_RATE)
    p_chase = base * 100

    # RRR pressure: every 1 run of extra RRR reduces win chance by ~5%
    p_chase -= rrr_vs_crr * 5

    # Wickets: each wicket lost reduces win chance
    wicket_penalty = {
        10: 0, 9: 2, 8: 4, 7: 6, 6: 9,
         5: 13, 4: 18, 3: 24, 2: 32, 1: 42,
    }
    p_chase -= wicket_penalty.get(wickets_remaining, 42)

    # H2H adjustment
    h2h_val = H2H.get((batting_team, bowling_team))
    if h2h_val is None:
        h2h_val = 1.0 - H2H.get((bowling_team, batting_team), OVERALL_CHASE_RATE)
    p_chase += (h2h_val - 0.50) * 10

    # Clamp
    p_chase = round(min(max(p_chase, 5.0), 95.0), 1)
    return p_chase, round(100.0 - p_chase, 1)


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
    batting_team = d.get("batting_team", "")
    bowling_team = d.get("bowling_team", "")
    venue        = d.get("venue", "")
    overs        = int(d.get("overs_completed", 10))
    curr_score   = int(d.get("current_score", 0))
    wickets_lost = int(d.get("wickets_lost", 0))

    if not all([batting_team, bowling_team, venue]):
        return jsonify({"error": "Please fill in all fields."}), 400
    if batting_team == bowling_team:
        return jsonify({"error": "Batting and bowling teams must be different."}), 400
    if curr_score <= 0:
        return jsonify({"error": "Current score must be greater than 0."}), 400
    if not (1 <= overs <= 19):
        return jsonify({"error": "Overs must be between 1 and 19."}), 400

    cricket_pred = cricket_aware_score(overs, curr_score, wickets_lost)

    if SCORE_MODEL is not None:
        feats = build_model_features(batting_team, bowling_team, venue,
                                     overs, curr_score, wickets_lost)
        cols  = list(SCORE_FEATS) if SCORE_FEATS else list(feats.keys())
        row   = pd.DataFrame([{c: feats.get(c, 0) for c in cols}])
        model_pred = int(round(float(SCORE_MODEL.predict(row)[0])))

        wih        = 10 - wickets_lost
        overs_rem  = 20 - overs
        overs_frac = min(overs_rem / 14.0, 1.0)
        max_add    = ABS_MAX_ADD_14OV.get(wih, 10) * overs_frac
        ceiling    = curr_score + max_add
        model_pred = int(round(min(model_pred, ceiling)))
        model_pred = max(model_pred, curr_score + 1)

        pred = int(round(0.60 * model_pred + 0.40 * cricket_pred))
        mode = "model"
    else:
        pred = cricket_pred
        mode = "demo"

    crr         = round(curr_score / overs, 2) if overs > 0 else 0
    overs_rem   = 20 - overs
    runs_needed = max(pred - curr_score, 0)
    req_rr      = round(runs_needed / overs_rem, 2) if overs_rem > 0 else 0
    quality_pct = int(QUALITY_REMAINING.get(int(min(wickets_lost, 9)), 0.02) * 100)
    phase       = 0 if overs <= 6 else (1 if overs <= 15 else 2)
    phase_name  = ["Powerplay", "Middle Overs", "Death Overs"][phase]

    return jsonify({
        "predicted_score": pred,
        "range_low":        max(pred - 8, curr_score + 1),
        "range_high":       pred + 8,
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
        df_in   = pd.DataFrame([{"team1": team1, "team2": team2, "venue": venue,
                                  "toss_winner": toss_winner,
                                  "toss_decision": toss_decision}])
        probas  = WIN_PIPELINE.predict_proba(df_in)[0]
        classes = list(WIN_PIPELINE.classes_)
        idx1    = classes.index(team1) if team1 in classes else None
        p1      = round(float(probas[idx1]) * 100, 1) if idx1 is not None else 50.0
        p2      = round(100 - p1, 1)
        mode    = "model"
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


# ─────────────────────────────────────────────────────────────────────────────
# NEW ROUTE: Live Win Probability (2nd Innings)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/predict/win-probability", methods=["POST"])
def predict_win_probability():
    d = request.get_json()

    batting_team    = d.get("batting_team", "")
    bowling_team    = d.get("bowling_team", "")
    venue           = d.get("venue", "")
    target          = int(d.get("target", 0))
    current_score   = int(d.get("current_score", 0))
    wickets_lost    = int(d.get("wickets_lost", 0))
    overs_completed = float(d.get("overs_completed", 6))

    # ── Validation ──
    if not all([batting_team, bowling_team, venue]):
        return jsonify({"error": "Please fill in all fields."}), 400
    if batting_team == bowling_team:
        return jsonify({"error": "Batting and bowling teams must be different."}), 400
    if target <= 0:
        return jsonify({"error": "Target must be greater than 0."}), 400
    if current_score < 0:
        return jsonify({"error": "Current score cannot be negative."}), 400
    if current_score >= target:
        return jsonify({"error": "Current score cannot exceed or equal the target."}), 400
    if not (1 <= overs_completed <= 19):
        return jsonify({"error": "Overs must be between 1 and 19."}), 400

    # ── Derived features ──
    runs_required     = max(target - current_score, 0)
    balls_remaining   = max(120 - overs_completed * 6, 1)
    required_run_rate = round(runs_required / (balls_remaining / 6), 4)
    current_run_rate  = round(current_score / overs_completed, 4) if overs_completed > 0 else 0
    rrr_vs_crr        = round(required_run_rate - current_run_rate, 4)
    wickets_remaining = 10 - wickets_lost

    if WIN_PROB_MODEL is not None and WIN_PROB_COLS is not None:
        # ── Look up venue and H2H from saved pkl tables ──
        v2_avg_val  = VENUE_2ND_AVG.get(venue, 180.0)
        v_chase_val = VENUE_CHASING_RATE.get(venue, OVERALL_CHASE_RATE)
        h2h_ratio   = OVERALL_CHASE_RATE

        if H2H_TABLE is not None:
            row_h2h = H2H_TABLE[
                (H2H_TABLE["batting_team"] == batting_team) &
                (H2H_TABLE["bowling_team"] == bowling_team)
            ]
            if len(row_h2h) > 0:
                h2h_ratio = float(row_h2h["h2h_win_ratio"].values[0])

        if VENUE_2ND_DF is not None:
            row_v2 = VENUE_2ND_DF[VENUE_2ND_DF["venue"] == venue]
            if len(row_v2) > 0:
                v2_avg_val = float(row_v2["venue_2nd_innings_avg"].values[0])

        if VENUE_CHASE_DF is not None:
            row_vc = VENUE_CHASE_DF[VENUE_CHASE_DF["venue"] == venue]
            if len(row_vc) > 0:
                v_chase_val = float(row_vc["venue_chase_win_rate"].values[0])

        input_df = pd.DataFrame([{
            "batting_team"         : batting_team,
            "bowling_team"         : bowling_team,
            "venue"                : venue,
            "overs_completed"      : overs_completed,
            "current_score"        : current_score,
            "wickets_lost"         : wickets_lost,
            "wickets_remaining"    : wickets_remaining,
            "target"               : target,
            "runs_required"        : runs_required,
            "balls_remaining"      : balls_remaining,
            "required_run_rate"    : required_run_rate,
            "current_run_rate"     : current_run_rate,
            "rrr_vs_crr"           : rrr_vs_crr,
            "venue_2nd_innings_avg": v2_avg_val,
            "venue_chase_win_rate" : v_chase_val,
            "h2h_win_ratio"        : h2h_ratio,
        }])

        input_df = pd.get_dummies(input_df,
                                  columns=["batting_team", "bowling_team", "venue"])
        input_df = input_df.reindex(columns=WIN_PROB_COLS, fill_value=0)

        prob     = WIN_PROB_MODEL.predict_proba(input_df)[0]
        p_chase  = round(float(prob[1]) * 100, 1)
        p_defend = round(float(prob[0]) * 100, 1)
        mode     = "model"

    else:
        p_chase, p_defend = demo_win_probability(
            batting_team, bowling_team, venue,
            target, current_score, wickets_lost, overs_completed
        )
        mode = "demo"

    winner     = batting_team if p_chase >= p_defend else bowling_team
    winner_pct = p_chase      if p_chase >= p_defend else p_defend
    conf = "High" if winner_pct >= 70 else ("Medium" if winner_pct >= 58 else "Low")

    return jsonify({
        "batting_team"      : batting_team,
        "bowling_team"      : bowling_team,
        "prob_chasing"      : p_chase,
        "prob_defending"    : p_defend,
        "winner"            : winner,
        "winner_pct"        : winner_pct,
        "confidence"        : conf,
        "runs_required"     : runs_required,
        "balls_remaining"   : int(balls_remaining),
        "required_run_rate" : required_run_rate,
        "current_run_rate"  : current_run_rate,
        "mode"              : mode,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)
