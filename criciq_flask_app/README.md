# CricIQ — IPL Analytics Platform

A Flask web application with two ML-powered features:

| Feature | Model | Metric |
|---------|-------|--------|
| Score Predictor | Random Forest (300 trees) | MAE 4.50 runs · R² 0.966 |
| Win Predictor | Logistic Regression | ~66% accuracy (5-fold CV) |

---

## Project Structure

```
criciq/
├── app.py                   ← Flask backend (routes + feature engineering)
├── requirements.txt
├── templates/
│   └── index.html           ← Single-page UI
├── static/
│   ├── css/style.css
│   └── js/app.js
│
│   ── Place your .pkl files here ──
├── ipl_score_model.pkl      ← Score model  (from score-prediction.ipynb cell 50)
├── model_columns.pkl        ← Feature list (18 features)
├── quality_remaining.pkl    ← Batting quality dict
├── ipl_encoders.pkl         ← Team/venue encoders
└── win_model.pkl            ← Win pipeline (from data_preprocessing.ipynb)
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy your .pkl files into this folder
#    (ipl_score_model.pkl, model_columns.pkl, quality_remaining.pkl,
#     ipl_encoders.pkl, win_model.pkl)

# 3. Start the server
python app.py

# 4. Open browser
http://localhost:5000
```

> **Without .pkl files**: The app runs in **demo mode** using the exact same
> feature-engineering logic from your notebooks, so predictions are realistic
> even before connecting the trained models.

---

## Connecting the Real Models

### Score Model
From `score-prediction.ipynb` (cell 50), save these files next to `app.py`:
```python
pickle.dump(model,             open("ipl_score_model.pkl",   "wb"))
pickle.dump(FEATURES,          open("model_columns.pkl",     "wb"))
pickle.dump(quality_remaining, open("quality_remaining.pkl", "wb"))
pickle.dump(encoders,          open("ipl_encoders.pkl",      "wb"))
```

### Win Model
From `data_preprocessing.ipynb`, save the trained pipeline:
```python
import pickle
# (after training the Logistic Regression pipeline on all features)
pickle.dump(model, open("win_model.pkl", "wb"))
```
The pipeline must expose `.predict_proba()` and have `.classes_` = list of team names.

---

## Features Used

### Score Predictor (18 features)
| Feature | Description |
|---------|-------------|
| `batting_team_strength` | Target-encoded mean batting score |
| `bowling_team_strength` | Target-encoded mean conceded score |
| `venue_first_innings_avg` | Historical 1st-innings avg at venue |
| `overs_completed` | Overs bowled so far |
| `current_score` | Runs scored so far |
| `wickets_lost` | Wickets fallen |
| `wickets_in_hand` | Remaining batters (10 − wickets) |
| `current_run_rate` | Runs ÷ overs |
| `crr_wicket_index` | CRR × (wickets_in_hand / 10) |
| `balls_per_wicket` | Balls faced per wicket lost |
| `balls_remaining` | (20 − overs) × 6 |
| `projected_score` | Naive CRR × 120 |
| `runs_to_venue_avg` | venue_avg − current_score |
| `game_phase` | 0=Powerplay / 1=Middle / 2=Death |
| `quality_batting_left` | Weighted batting depth remaining |
| `wicket_adj_projection` | Quality-weighted projection |
| `wicket_pressure` | wickets / overs |
| `rr_vs_phase_norm` | CRR − phase average RR |

### Win Predictor (from data_preprocessing.ipynb)
- `team1`, `team2`, `venue`, `toss_winner`, `toss_decision`
- `team1_win_pct`, `team2_win_pct`
- `head_to_head_win_pct`
- `team1_venue_win_pct`
- `team1_home`, `team2_home`
- `venue_chasing_rate`
- `team1_won_toss`, `toss_decision_field`, `team1_chasing`
- `team1_last7_form`, `team2_last7_form`
