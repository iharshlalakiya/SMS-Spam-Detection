"""
SMS Spam Detection — Streamlit Web App
=======================================
Run with:
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

import joblib
import streamlit as st

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sms_spam.data.preprocessing import preprocess_text
from sms_spam.models.svm import SpamDetector
from sms_spam.features.extraction import TFIDFExtractor  # noqa: F401 — required for pickle to deserialize tfidf_vectorizer.pkl
from sms_spam.logs.logger import get_logger
from monitoring.app_metrics import start_metrics_server, record_prediction, record_error, set_model_loaded

start_metrics_server(9300)

log = get_logger("sms_spam.app", log_to_console=False)

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# Custom CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main background ── */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 0;
}

/* ── Result cards ── */
.result-spam {
    background: linear-gradient(135deg, #3b0764, #7c1b1b);
    border: 1px solid #ef4444;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: pulse 0.5s ease-in-out;
}
.result-ham {
    background: linear-gradient(135deg, #052e16, #0c2340);
    border: 1px solid #22c55e;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: pulse 0.5s ease-in-out;
}
@keyframes pulse {
    0%   { transform: scale(0.96); opacity: 0; }
    100% { transform: scale(1);    opacity: 1; }
}
.result-emoji { font-size: 3.5rem; }
.result-label {
    font-size: 2rem;
    font-weight: 800;
    margin: 0.4rem 0;
}
.result-sub { color: #cbd5e1; font-size: 0.95rem; }

/* ── Confidence bar ── */
.conf-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.3rem;
    color: #e2e8f0;
    font-size: 0.9rem;
}
.conf-bar-bg {
    background: #1e293b;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin-bottom: 0.8rem;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* ── Stat cards ── */
.stat-grid {
    display: flex;
    gap: 1rem;
    margin-top: 0.5rem;
}
.stat-card {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #a78bfa;
}
.stat-label {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(15, 15, 26, 0.95);
    border-right: 1px solid rgba(255,255,255,0.07);
}

/* ── Textarea ── */
textarea {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    background: rgba(255,255,255,0.04) !important;
    color: #f1f5f9 !important;
    font-size: 1rem !important;
}
textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.2) !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
    color: white !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.25s ease !important;
    letter-spacing: 0.03em;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(124,58,237,0.45) !important;
}

/* ── History items ── */
.hist-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}
.hist-spam { border-left: 3px solid #ef4444; }
.hist-ham  { border-left: 3px solid #22c55e; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Load model (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    models_dir = ROOT / "models"
    svm_path   = models_dir / "svm.pkl"
    tfidf_path = models_dir / "tfidf_vectorizer.pkl"

    if not svm_path.exists() or not tfidf_path.exists():
        return None, None

    detector = SpamDetector()
    detector.load(str(svm_path))
    tfidf = joblib.load(str(tfidf_path))
    log.info("Model loaded successfully from %s", models_dir)
    set_model_loaded(True)
    return detector, tfidf


# ══════════════════════════════════════════════════════════════════════════════
# Predict helper
# ══════════════════════════════════════════════════════════════════════════════
def predict(text: str, detector, tfidf):
    cleaned   = preprocess_text(text)
    vectorized = tfidf.transform([cleaned])
    label      = detector.predict(vectorized)[0]
    proba      = detector.predict_proba(vectorized)[0]
    spam_prob  = float(proba[1]) * 100
    ham_prob   = float(proba[0]) * 100
    return int(label), spam_prob, ham_prob


# ══════════════════════════════════════════════════════════════════════════════
# Session state init
# ══════════════════════════════════════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🛡️ SMS Spam Detector")
    st.markdown("---")
    st.markdown("**Model:** Linear SVM")
    st.markdown("**Vectorizer:** TF-IDF (5 000 features)")

    st.markdown("---")
    st.markdown("#### 📊 Session Stats")

    total = len(st.session_state.history)
    spam_count = sum(1 for h in st.session_state.history if h["label"] == 1)
    ham_count  = total - spam_count

    col1, col2 = st.columns(2)
    col1.metric("Total Checked", total)
    col2.metric("Spam Found",    spam_count)

    st.markdown("---")
    st.markdown("#### 🧪 Example messages")
    examples = {
        "💰 Prize winner":   "CONGRATULATIONS! You've won a £1,000 prize. Call 0800 123456 now!",
        "📅 Meeting":        "Hey, are we still on for the meeting at 3pm today?",
        "🏦 Bank alert":     "URGENT: Your account has been suspended. Verify at http://fake-bank.com",
        "👋 Casual":         "What time are you coming home tonight?",
        "🎁 Free offer":     "FREE entry in 2 a weekly comp to win FA Cup Final tkts! Text FA to 87121",
    }
    for label, msg in examples.items():
        if st.button(label, key=f"ex_{label}"):
            st.session_state["input_text"] = msg

    if st.button("🗑️ Clear History", key="clear_hist"):
        st.session_state.history = []

    st.markdown("---")
    st.caption("Built with ❤️ using scikit-learn + Streamlit")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

# Hero
st.markdown("""
<div class="hero">
    <h1>🛡️ SMS Spam Detector</h1>
    <p>Powered by a Linear SVM trained on 5 574 real SMS messages</p>
</div>
""", unsafe_allow_html=True)

# Load model
detector, tfidf = load_model()
if detector is None:
    st.error(
        "⚠️ **Model not found.** Please train the model first by running:\n\n"
        "```bash\npython main.py\n```"
    )
    st.stop()

# Input area
user_input = st.text_area(
    "✉️ Paste or type your SMS message below:",
    value=st.session_state.get("input_text", ""),
    height=130,
    placeholder="e.g. Congratulations! You've won a free prize. Click here to claim...",
    key="main_input",
)

col_btn, col_clear = st.columns([3, 1])
with col_btn:
    analyse = st.button("🔍 Analyse Message", key="analyse_btn")
with col_clear:
    if st.button("✕ Clear", key="clear_btn"):
        st.session_state["input_text"] = ""
        st.rerun()

# ── Result ────────────────────────────────────────────────────────────────────
if analyse:
    text = user_input.strip()
    if not text:
        st.warning("⚠️ Please enter a message first.")
    else:
        with st.spinner("Analysing..."):
            time.sleep(0.3)   # tiny delay for UX feel
            t0 = time.time()
            label, spam_prob, ham_prob = predict(text, detector, tfidf)
            record_prediction(label, time.time() - t0)
            log.info("Prediction: %s (spam=%.1f%%) | input=%r", "SPAM" if label else "HAM", spam_prob, text[:60])

        # Result card
        if label == 1:
            st.markdown(f"""
            <div class="result-spam">
                <div class="result-emoji">🚨</div>
                <div class="result-label" style="color:#ef4444;">SPAM</div>
                <div class="result-sub">This message is likely spam. Be careful!</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-ham">
                <div class="result-emoji">✅</div>
                <div class="result-label" style="color:#22c55e;">LEGITIMATE</div>
                <div class="result-sub">This message appears to be safe.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence bars
        st.markdown("#### 📊 Confidence")
        spam_color = "#ef4444" if label == 1 else "#64748b"
        ham_color  = "#22c55e" if label == 0 else "#64748b"

        st.markdown(f"""
        <div class="conf-row"><span>🚨 Spam</span><span><b>{spam_prob:.1f}%</b></span></div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{spam_prob:.1f}%;background:{spam_color};"></div>
        </div>
        <div class="conf-row"><span>✅ Ham (Legitimate)</span><span><b>{ham_prob:.1f}%</b></span></div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{ham_prob:.1f}%;background:{ham_color};"></div>
        </div>
        """, unsafe_allow_html=True)

        # Save to history
        st.session_state.history.insert(0, {
            "text":      text[:80] + ("…" if len(text) > 80 else ""),
            "label":     label,
            "spam_prob": spam_prob,
            "ham_prob":  ham_prob,
        })

# ── History ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("#### 🕓 Recent Checks")
    for i, h in enumerate(st.session_state.history[:8]):
        cls   = "hist-spam" if h["label"] == 1 else "hist-ham"
        badge = "🚨 SPAM" if h["label"] == 1 else "✅ HAM"
        conf  = h["spam_prob"] if h["label"] == 1 else h["ham_prob"]
        st.markdown(f"""
        <div class="hist-item {cls}">
            <span style="font-weight:600;">{badge}</span>
            <span style="color:#64748b;font-size:0.85rem;margin-left:0.5rem;">{conf:.1f}% confidence</span>
            <div style="color:#cbd5e1;font-size:0.9rem;margin-top:0.3rem;">{h['text']}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#334155;font-size:0.8rem;margin-top:3rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.05);">
    SMS Spam Detector &nbsp;·&nbsp; Linear SVM &nbsp;·&nbsp; Accuracy 97.94% &nbsp;·&nbsp; F1 0.9181
</div>
""", unsafe_allow_html=True)
