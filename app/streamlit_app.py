import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Score Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f0f1a; }
    #MainMenu, footer, header { visibility: hidden; }

    .section-title {
        color: #a78bfa;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .metric-card {
        background: #0d0d1f;
        border: 1px solid #2d2d5e;
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
    }
    .metric-label {
        color: #6b7280;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #f1f5f9;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .metric-sub {
        color: #6b7280;
        font-size: 0.72rem;
        margin-top: 2px;
    }
    .badge-powerplay {
        background:#1e3a5f; color:#60a5fa; border:1px solid #3b82f6;
        padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600;
    }
    .badge-middle {
        background:#1e3a1e; color:#4ade80; border:1px solid #22c55e;
        padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600;
    }
    .badge-death {
        background:#3a1e1e; color:#f87171; border:1px solid #ef4444;
        padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600;
    }
    .pred-box {
        background: linear-gradient(135deg, #065f46 0%, #064e3b 100%);
        border: 2px solid #10b981;
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        margin: 8px 0;
    }
    .pred-label {
        color: #6ee7b7; font-size: 0.85rem; font-weight: 600;
        letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px;
    }
    .pred-score { color: #ffffff; font-size: 3.5rem; font-weight: 800; line-height: 1; }
    .pred-range { color: #a7f3d0; font-size: 1rem; margin-top: 10px; }
    .constraint-box {
        background: #1c1200;
        border: 1px solid #d97706;
        border-radius: 10px;
        padding: 10px 16px;
        color: #fbbf24;
        font-size: 0.82rem;
        margin-top: 8px;
    }
    .quality-bar-bg {
        background: #1f2937;
        border-radius: 8px;
        height: 8px;
        margin-top: 6px;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        padding: 14px !important;
        letter-spacing: 1px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model             = pickle.load(open("./notebooks/ipl_score_model.pkl",   "rb"))
    FEATURES          = pickle.load(open("./notebooks/model_columns.pkl",     "rb"))
    quality_remaining = pickle.load(open("./notebooks/quality_remaining.pkl", "rb"))
    encoders          = pickle.load(open("./notebooks/ipl_encoders.pkl",      "rb"))
    return model, FEATURES, quality_remaining, encoders

model, FEATURES, quality_remaining, encoders = load_artifacts()

TEAMS       = encoders["teams"]
VENUES      = encoders["venues"]
batting_enc = encoders["batting_strength"]
bowling_enc = encoders["bowling_strength"]
venue_enc   = encoders["venue_avg"]
global_avg  = encoders.get("global_avg", 165.0)


# ─────────────────────────────────────────────────────────────────────────────
# CRICKET CONSTRAINT ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def apply_cricket_constraints(raw_prediction, current_score,
                               overs_completed, wickets_lost,
                               quality_batting_left):

    balls_remaining = (20 - overs_completed) * 6
    overs_remaining = balls_remaining / 6
    wickets_in_hand = 10 - wickets_lost
    prediction      = raw_prediction

    # 1. Never below current score
    prediction = max(prediction, current_score)

    # 2. Absolute max: 3.5 runs per ball remaining
    prediction = min(prediction, current_score + balls_remaining * 3.5)

    # 3. Last over: max 28 additional runs
    if overs_remaining <= 1:
        prediction = min(prediction, current_score + 28)

    # 4. Tail-ender cap: wickets in hand ≤ 4
    if wickets_in_hand <= 4:
        tail_rr = {4: 7.0, 3: 5.5, 2: 4.5, 1: 3.5}.get(wickets_in_hand, 3.5)
        prediction = min(prediction, current_score + overs_remaining * tail_rr)

    # 5. Wicket collapse penalty: quality < 30% with many overs left
    if quality_batting_left < 0.30 and overs_remaining > 4:
        collapse_cap = current_score + (overs_remaining * 9.0 * quality_batting_left * 2.5)
        prediction = min(prediction, collapse_cap)

    # 6. Minimum floor
    min_floor = current_score + (overs_remaining * 4.0 * max(quality_batting_left, 0.1))
    prediction = max(prediction, min_floor)

    return round(prediction)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_score(batting_team, bowling_team, venue,
                  overs_completed, current_score, wickets_lost):

    balls_completed  = overs_completed * 6
    balls_remaining  = 120 - balls_completed
    wickets_in_hand  = 10 - wickets_lost
    crr              = current_score / overs_completed if overs_completed > 0 else 0.0

    batting_strength  = batting_enc.get(batting_team, global_avg)
    bowling_strength  = bowling_enc.get(bowling_team, global_avg)
    venue_avg         = venue_enc.get(venue, global_avg)

    quality_left      = quality_remaining.get(int(wickets_lost), 0.0)

    if overs_completed <= 6:    game_phase = 0
    elif overs_completed <= 15: game_phase = 1
    else:                       game_phase = 2

    phase_avg_rr = {0: 8.5, 1: 8.0, 2: 10.5}

    row = {
        "batting_team_strength":   batting_strength,
        "bowling_team_strength":   bowling_strength,
        "venue_first_innings_avg": venue_avg,
        "overs_completed":         overs_completed,
        "current_score":           current_score,
        "wickets_lost":            wickets_lost,
        "wickets_in_hand":         wickets_in_hand,
        "current_run_rate":        crr,
        "crr_wicket_index":        crr * (wickets_in_hand / 10),
        "balls_per_wicket":        balls_completed / (wickets_lost + 1),
        "balls_remaining":         balls_remaining,
        "projected_score":         (current_score / balls_completed * 120) if balls_completed > 0 else 0.0,
        "runs_to_venue_avg":       venue_avg - current_score,
        "game_phase":              game_phase,
        "quality_batting_left":    quality_left,
        "wicket_adj_projection":   current_score + (balls_remaining / 6) * crr * quality_left,
        "wicket_pressure":         wickets_lost / (overs_completed + 0.1),
        "rr_vs_phase_norm":        crr - phase_avg_rr[game_phase],
    }

    raw_pred   = float(model.predict(pd.DataFrame([row])[FEATURES])[0])
    final_pred = apply_cricket_constraints(
        raw_pred, current_score, overs_completed, wickets_lost, quality_left
    )
    return final_pred, round(raw_pred), quality_left, venue_avg


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:24px 0 8px 0;">
    <div style="font-size:2.6rem; font-weight:800; color:#f1f5f9; letter-spacing:-1px;">
        🏏 IPL Score Predictor
    </div>
    <div style="color:#6b7280; font-size:0.9rem; margin-top:6px; letter-spacing:1px;">
        XGBoost · Phase-Aware · Wicket-Quality Engine · Constraint Corrected
    </div>
</div>
<hr style="border:none; border-top:1px solid #2d2d5e; margin:16px 0 24px 0;">
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 1], gap="large")

with left_col:

    # ── Match Setup ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">⚔️ Match Setup</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        batting_team = st.selectbox("🏏 Batting Team", TEAMS)
    with c2:
        bowling_team = st.selectbox("🎯 Bowling Team",
                                    [t for t in TEAMS if t != batting_team])

    venue = st.selectbox("🏟️ Venue", VENUES)
    st.markdown("<hr style='border:none;border-top:1px solid #2d2d5e;margin:16px 0'>",
                unsafe_allow_html=True)

    # ── Match State ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Current Match State</div>',
                unsafe_allow_html=True)

    # Ball-level overs input
    st.markdown('<p style="color:#94a3b8;font-size:0.82rem;font-weight:600;'
                'margin-bottom:4px;">⏱️ Overs Completed</p>', unsafe_allow_html=True)
    ov1, ov2 = st.columns(2)
    with ov1:
        overs_int   = st.number_input("Overs", min_value=6, max_value=19,
                                      value=10, step=1, label_visibility="collapsed")
    with ov2:
        balls_extra = st.selectbox("Balls", [0,1,2,3,4,5],
                                   format_func=lambda x: f".{x} balls",
                                   label_visibility="collapsed")

    overs = overs_int + balls_extra / 6
    st.caption(f"Overs: **{overs_int}.{balls_extra}** ({overs:.3f} decimal overs)")

    sc_col, wk_col = st.columns(2)
    with sc_col:
        score   = st.number_input("🏃 Current Score", min_value=0,
                                   max_value=350, value=95, step=1)
    with wk_col:
        wickets = st.slider("💀 Wickets Lost", min_value=0, max_value=9, value=2)

    # ── Live Context ──────────────────────────────────────────────────────────
    crr          = score / overs if overs > 0 else 0.0
    wickets_left = 10 - wickets
    balls_left   = int((20 - overs) * 6)
    venue_avg_d  = venue_enc.get(venue, global_avg)
    quality      = quality_remaining.get(int(wickets), 0.0)

    if overs <= 6:    phase_label = "Powerplay"; phase_badge = "badge-powerplay"
    elif overs <= 15: phase_label = "Middle";     phase_badge = "badge-middle"
    else:             phase_label = "Death";      phase_badge = "badge-death"

    st.markdown("<hr style='border:none;border-top:1px solid #2d2d5e;margin:16px 0'>",
                unsafe_allow_html=True)
    st.markdown('<div class="section-title">📡 Live Match Context</div>',
                unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">⚡ Run Rate</div>'
                    f'<div class="metric-value">{crr:.2f}</div>'
                    f'<div class="metric-sub">runs/over</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">🏏 Wickets Left</div>'
                    f'<div class="metric-value">{wickets_left}</div>'
                    f'<div class="metric-sub">in hand</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">🎯 Balls Left</div>'
                    f'<div class="metric-value">{balls_left}</div>'
                    f'<div class="metric-sub">{20-overs:.1f} overs</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    m4, m5, m6 = st.columns(3)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">📍 Venue Avg</div>'
                    f'<div class="metric-value">{venue_avg_d:.0f}</div>'
                    f'<div class="metric-sub">1st innings</div></div>', unsafe_allow_html=True)
    with m5:
        st.markdown(f'<div class="metric-card"><div class="metric-label">💪 Batting Quality</div>'
                    f'<div class="metric-value">{quality*100:.0f}%</div>'
                    f'<div class="metric-sub">remaining</div></div>', unsafe_allow_html=True)
    with m6:
        st.markdown(f'<div class="metric-card"><div class="metric-label">🕹️ Phase</div>'
                    f'<div class="metric-value" style="font-size:1rem;padding-top:6px;">'
                    f'<span class="{phase_badge}">{phase_label}</span></div>'
                    f'<div class="metric-sub">&nbsp;</div></div>', unsafe_allow_html=True)

    # Quality bar
    q_color = "#4ade80" if quality > 0.5 else ("#facc15" if quality > 0.25 else "#f87171")
    st.markdown(f"""
    <div style="margin-top:14px;">
        <div style="display:flex;justify-content:space-between;
                    color:#6b7280;font-size:0.72rem;margin-bottom:4px;">
            <span>Batting Quality Remaining</span><span>{quality*100:.0f}%</span>
        </div>
        <div class="quality-bar-bg">
            <div style="width:{quality*100:.0f}%;background:{q_color};
                        height:8px;border-radius:8px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🔮  PREDICT FINAL SCORE", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN
# ─────────────────────────────────────────────────────────────────────────────
with right_col:

    st.markdown('<div class="section-title">🏆 Prediction Result</div>',
                unsafe_allow_html=True)

    if predict_btn:

        predicted, raw_pred, q_left, v_avg = predict_score(
            batting_team, bowling_team, venue, overs, score, wickets
        )
        low  = int(predicted * 0.94)
        high = int(predicted * 1.06)
        constraint_fired = (predicted != raw_pred)

        # Main result
        st.markdown(f"""
        <div class="pred-box">
            <div class="pred-label">🏆 Predicted Final Score</div>
            <div class="pred-score">{predicted}</div>
            <div class="pred-range">📊 Likely Range: {low} – {high}</div>
        </div>
        """, unsafe_allow_html=True)

        if constraint_fired:
            st.markdown(f"""
            <div class="constraint-box">
                ⚠️ ML model predicted <b>{raw_pred}</b> — cricket logic
                adjusted to <b>{predicted}</b>
                (batting quality: <b>{q_left*100:.0f}%</b> remaining)
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Drivers
        st.markdown('<div class="section-title">🔍 Prediction Drivers</div>',
                    unsafe_allow_html=True)

        naive_proj   = round(score + (balls_left / 6) * crr)
        quality_proj = round(score + (balls_left / 6) * crr * q_left)
        vs_venue     = predicted - v_avg
        vs_color     = "#4ade80" if vs_venue >= 0 else "#f87171"
        vs_sign      = "+" if vs_venue >= 0 else ""

        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">📈 Naive Projection</div>'
                        f'<div class="metric-value" style="color:#94a3b8;">{naive_proj}</div>'
                        f'<div class="metric-sub">CRR × overs left</div></div>',
                        unsafe_allow_html=True)
        with d2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">🎯 Quality Projection</div>'
                        f'<div class="metric-value" style="color:#a78bfa;">{quality_proj}</div>'
                        f'<div class="metric-sub">Wicket-adjusted</div></div>',
                        unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        d3, d4 = st.columns(2)
        with d3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">🏟️ Venue Average</div>'
                        f'<div class="metric-value" style="color:#60a5fa;">{v_avg:.0f}</div>'
                        f'<div class="metric-sub">IPL 2023–25</div></div>',
                        unsafe_allow_html=True)
        with d4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">📊 vs Venue Avg</div>'
                        f'<div class="metric-value" style="color:{vs_color};">'
                        f'{vs_sign}{vs_venue:.0f}</div>'
                        f'<div class="metric-sub">above/below</div></div>',
                        unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # Wicket impact
        st.markdown('<div class="section-title">📉 Wicket Impact Analysis</div>',
                    unsafe_allow_html=True)
        st.caption(f"Same score ({score}) & overs ({overs_int}.{balls_extra}) — effect of each wicket")

        rows = []
        for w in range(10):
            p, rp, q, _ = predict_score(
                batting_team, bowling_team, venue, overs, score, w
            )
            rows.append({
                "Wkts Lost":  w,
                "Wkts Left":  10 - w,
                "Quality":    f"{quality_remaining.get(w,0)*100:.0f}%",
                "ML Pred":    rp,
                "Final Pred": p,
                "":           "◀" if w == wickets else ""
            })

        impact_df = pd.DataFrame(rows)

        def highlight_row(row):
            if row[""] == "◀":
                return ["background-color:#1e3a5f; color:white"] * len(row)
            return [""] * len(row)

        st.dataframe(
            impact_df.style.apply(highlight_row, axis=1),
            use_container_width=True,
            hide_index=True,
            height=370
        )

        st.bar_chart(
            impact_df.set_index("Wkts Lost")[["Final Pred"]].rename(
                columns={"Final Pred": "Predicted Score"}
            ),
            color="#7c3aed"
        )

    else:
        st.markdown("""
        <div style="border:2px dashed #2d2d5e; border-radius:20px;
                    padding:60px 30px; text-align:center; color:#374151;">
            <div style="font-size:3rem; margin-bottom:16px;">🏏</div>
            <div style="color:#6b7280; font-size:1rem; font-weight:600;">
                Set match conditions on the left<br>
                and click <b style="color:#7c3aed">Predict</b> to see results
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">💡 Example Scenarios</div>',
                    unsafe_allow_html=True)

        for title, scenario, expected in [
            ("🔥 Strong Position",  "10 overs, 95/2",  "~185–200"),
            ("⚠️ Batting Collapse", "10 overs, 60/7",  "~90–110"),
            ("💀 Death Overs",      "18 overs, 170/4", "~200–215"),
            ("🐢 Low Scoring",      "10 overs, 50/5",  "~100–120"),
        ]:
            st.markdown(f"""
            <div style="background:#111827; border:1px solid #1f2937;
                        border-radius:10px; padding:10px 14px; margin-bottom:8px;
                        display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="color:#e2e8f0; font-size:0.85rem; font-weight:600;">
                        {title}
                    </div>
                    <div style="color:#6b7280; font-size:0.75rem;">{scenario}</div>
                </div>
                <div style="color:#a78bfa; font-weight:700; font-size:0.9rem;">
                    {expected}
                </div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="border:none;border-top:1px solid #1f2937;margin:32px 0 16px 0;">
<div style="text-align:center;color:#374151;font-size:0.75rem;padding-bottom:16px;">
    IPL Score Predictor · XGBoost Model · Trained on IPL 2023–2025 · Final Year Project
</div>
""", unsafe_allow_html=True)