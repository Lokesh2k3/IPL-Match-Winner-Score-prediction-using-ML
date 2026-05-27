/* ═══════════════════════════════════════════════════════════
   CricIQ  ·  app.js
   Navigation + Score Predictor + Win Predictor + Win Probability
═══════════════════════════════════════════════════════════ */

// ── Helpers ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const show = id => $(id).classList.remove('hidden');
const hide = id => $(id).classList.add('hidden');

// ── Tab Navigation ────────────────────────────────────────────────────────────
function switchPage(target) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-card, .hnav-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.target === target);
  });
  const pg = document.getElementById('page-' + target);
  if (pg) pg.classList.add('active');
}

document.querySelectorAll('.tab-card, .hnav-btn').forEach(btn => {
  btn.addEventListener('click', () => switchPage(btn.dataset.target));
});

// ── Auto-fill Venue Avg ───────────────────────────────────────────────────────
function autoFillVenueAvg() {
  // no extra field in this version — venue avg sent internally
}

// ── Toss sync ─────────────────────────────────────────────────────────────────
function syncToss() {
  const t1  = $('w-team1').value;
  const t2  = $('w-team2').value;
  const sel = $('w-toss');
  sel.innerHTML = '<option value="">Select…</option>';
  if (t1) sel.innerHTML += `<option value="${t1}">${t1}</option>`;
  if (t2) sel.innerHTML += `<option value="${t2}">${t2}</option>`;
}

// ── Team abbreviation helper ──────────────────────────────────────────────────
function teamAbbr(name) {
  return name.split(' ').map(w => w[0]).join('').slice(0, 3).toUpperCase();
}

// ─────────────────────────────────────────────────────────────────────────────
// SCORE PREDICTOR
// ─────────────────────────────────────────────────────────────────────────────
async function submitScore() {
  const batting = $('s-batting').value;
  const bowling = $('s-bowling').value;
  const venue   = $('s-venue').value;
  const overs   = parseFloat($('s-overs').value);
  const score   = parseInt($('s-score').value) || 0;
  const wickets = parseInt($('s-wickets').value) || 0;

  if (!batting || !bowling || !venue) {
    return alert('Please select batting team, bowling team and venue.');
  }
  if (batting === bowling) {
    return alert('Batting and bowling teams must be different.');
  }
  if (score <= 0) {
    return alert('Please enter the current score.');
  }

  hide('score-idle'); hide('score-result');
  show('score-loading');

  try {
    const res = await fetch('/predict/score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        batting_team: batting, bowling_team: bowling,
        venue, overs_completed: overs,
        current_score: score, wickets_lost: wickets,
      }),
    });
    const data = await res.json();

    if (data.error) { alert(data.error); hide('score-loading'); show('score-idle'); return; }

    hide('score-loading');
    renderScore(data);

  } catch (err) {
    console.error(err);
    alert('Server error. Is Flask running on port 5000?');
    hide('score-loading'); show('score-idle');
  }
}

function renderScore(d) {
  $('r-score').textContent = d.predicted_score;
  $('r-range').textContent = `Expected range: ${d.range_low} – ${d.range_high} runs`;
  $('r-crr').textContent   = d.current_run_rate;
  $('r-rem').textContent   = d.overs_remaining;
  $('r-runs').textContent  = d.runs_needed;
  $('r-reqrr').textContent = d.required_rr;

  $('r-quality-pct').textContent = d.quality_pct + '%';
  $('r-phase').textContent       = '📍 ' + d.phase;

  // Animate quality bar
  const bar = $('r-quality-bar');
  bar.style.width = '0';
  requestAnimationFrame(() => {
    setTimeout(() => { bar.style.width = d.quality_pct + '%'; }, 50);
  });

  // Colour CRR
  const crrEl = $('r-crr');
  crrEl.style.color = d.current_run_rate >= 9 ? 'var(--green)' :
                      d.current_run_rate >= 7 ? 'var(--amber)' : 'var(--red)';

  show('score-result');
}

// ─────────────────────────────────────────────────────────────────────────────
// WIN PREDICTOR  (Pre-match)
// ─────────────────────────────────────────────────────────────────────────────
async function submitWin() {
  const team1    = $('w-team1').value;
  const team2    = $('w-team2').value;
  const venue    = $('w-venue').value;
  const toss     = $('w-toss').value;
  const decision = $('w-decision').value;

  if (!team1 || !team2 || !venue || !toss) {
    return alert('Please fill in all fields.');
  }
  if (team1 === team2) {
    return alert('Please select two different teams.');
  }

  hide('win-idle'); hide('win-result');
  show('win-loading');

  try {
    const res = await fetch('/predict/win', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        team1, team2, venue,
        toss_winner: toss, toss_decision: decision,
      }),
    });
    const data = await res.json();

    if (data.error) { alert(data.error); hide('win-loading'); show('win-idle'); return; }

    hide('win-loading');
    renderWin(data);

  } catch (err) {
    console.error(err);
    alert('Server error. Is Flask running on port 5000?');
    hide('win-loading'); show('win-idle');
  }
}

function renderWin(d) {
  const isT1Winner = d.prob_team1 >= d.prob_team2;

  // Winner banner
  $('w-winner').textContent = d.winner;

  // Confidence chip
  const chip = $('w-conf');
  chip.textContent = d.confidence + ' Confidence';
  chip.className = 'conf-chip';
  if (d.confidence === 'Medium') chip.classList.add('medium');
  else if (d.confidence === 'Low') chip.classList.add('low');

  // Team 1
  $('wc1').textContent = teamAbbr(d.team1);
  $('wn1').textContent = d.team1;
  $('wp1').textContent = d.prob_team1 + '%';
  $('wc1').className   = 'team-chip ' + (isT1Winner ? 'winner' : 'loser');
  $('wp1').className   = 'prob-num '  + (isT1Winner ? 'winner' : 'loser');
  $('pb1').className   = 'prob-fill ' + (isT1Winner ? 'winner' : 'loser');

  // Team 2
  $('wc2').textContent = teamAbbr(d.team2);
  $('wn2').textContent = d.team2;
  $('wp2').textContent = d.prob_team2 + '%';
  $('wc2').className   = 'team-chip ' + (!isT1Winner ? 'winner' : 'loser');
  $('wp2').className   = 'prob-num '  + (!isT1Winner ? 'winner' : 'loser');
  $('pb2').className   = 'prob-fill ' + (!isT1Winner ? 'winner' : 'loser');

  // Animate bars
  $('pb1').style.width = '0';
  $('pb2').style.width = '0';
  requestAnimationFrame(() => {
    setTimeout(() => {
      $('pb1').style.width = d.prob_team1 + '%';
      $('pb2').style.width = d.prob_team2 + '%';
    }, 60);
  });

  // Confidence dots (10 dots = 100%)
  const dots = $('conf-dots');
  dots.innerHTML = '';
  const filled = Math.round(d.winner_pct / 10);
  for (let i = 0; i < 10; i++) {
    const el = document.createElement('div');
    el.className = 'conf-dot' + (i < filled ? ' on' : '');
    dots.appendChild(el);
  }

  show('win-result');
}

// ─────────────────────────────────────────────────────────────────────────────
// WIN PROBABILITY  (Live 2nd Innings Chase)
// ─────────────────────────────────────────────────────────────────────────────
async function submitWinProb() {
  const batting_team    = $('wp-batting').value;
  const bowling_team    = $('wp-bowling').value;
  const venue           = $('wp-venue').value;
  const target          = parseInt($('wp-target').value) || 0;
  const overs_completed = parseInt($('wp-overs').value);
  const current_score   = parseInt($('wp-score').value) || 0;
  const wickets_lost    = parseInt($('wp-wickets').value) || 0;

  if (!batting_team || !bowling_team || !venue) {
    return alert('Please select batting team, bowling team and venue.');
  }
  if (batting_team === bowling_team) {
    return alert('Batting and bowling teams must be different.');
  }
  if (target <= 0) {
    return alert('Please enter a valid target score.');
  }
  if (current_score >= target) {
    return alert('Current score cannot be equal to or exceed the target.');
  }

  hide('winprob-idle'); hide('winprob-result');
  show('winprob-loading');

  try {
    const res = await fetch('/predict/win-probability', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        batting_team, bowling_team, venue,
        target, current_score, wickets_lost, overs_completed,
      }),
    });
    const data = await res.json();

    if (data.error) { alert(data.error); hide('winprob-loading'); show('winprob-idle'); return; }

    hide('winprob-loading');
    renderWinProb(data);

  } catch (err) {
    console.error(err);
    alert('Server error. Is Flask running on port 5000?');
    hide('winprob-loading'); show('winprob-idle');
  }
}

function renderWinProb(d) {
  const isChaserWinner = d.prob_chasing >= d.prob_defending;

  // Winner banner
  $('wp-winner').textContent = d.winner;

  // Confidence chip
  const chip = $('wp-conf');
  chip.textContent = d.confidence + ' Confidence';
  chip.className = 'conf-chip';
  if (d.confidence === 'Medium') chip.classList.add('medium');
  else if (d.confidence === 'Low') chip.classList.add('low');

  // Chasing team (batting)
  $('wpc1').textContent = teamAbbr(d.batting_team);
  $('wpn1').textContent = d.batting_team;
  $('wpp1').textContent = d.prob_chasing + '%';
  $('wpc1').className   = 'team-chip ' + (isChaserWinner ? 'winner' : 'loser');
  $('wpp1').className   = 'prob-num '  + (isChaserWinner ? 'winner' : 'loser');
  $('wpb1').className   = 'prob-fill ' + (isChaserWinner ? 'winner' : 'loser');

  // Defending team (bowling)
  $('wpc2').textContent = teamAbbr(d.bowling_team);
  $('wpn2').textContent = d.bowling_team;
  $('wpp2').textContent = d.prob_defending + '%';
  $('wpc2').className   = 'team-chip ' + (!isChaserWinner ? 'winner' : 'loser');
  $('wpp2').className   = 'prob-num '  + (!isChaserWinner ? 'winner' : 'loser');
  $('wpb2').className   = 'prob-fill ' + (!isChaserWinner ? 'winner' : 'loser');

  // Animate bars
  $('wpb1').style.width = '0';
  $('wpb2').style.width = '0';
  requestAnimationFrame(() => {
    setTimeout(() => {
      $('wpb1').style.width = d.prob_chasing + '%';
      $('wpb2').style.width = d.prob_defending + '%';
    }, 60);
  });

  // Chase stats grid
  $('wp-r-req').textContent   = d.runs_required;
  $('wp-r-balls').textContent = d.balls_remaining;
  $('wp-r-rrr').textContent   = d.required_run_rate;
  $('wp-r-crr').textContent   = d.current_run_rate;

  // Colour RRR — red if pressure is high
  const rrrEl = $('wp-r-rrr');
  const rrr   = parseFloat(d.required_run_rate);
  rrrEl.style.color = rrr >= 12 ? 'var(--red)' :
                      rrr >= 9  ? 'var(--amber)' : 'var(--green)';

  // Confidence dots (10 dots = 100%)
  const dots = $('wp-conf-dots');
  dots.innerHTML = '';
  const filled = Math.round(d.winner_pct / 10);
  for (let i = 0; i < 10; i++) {
    const el = document.createElement('div');
    el.className = 'conf-dot' + (i < filled ? ' on' : '');
    dots.appendChild(el);
  }

  show('winprob-result');
}
