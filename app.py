# app.py
"""
Smart Cyber-Safety Analyzer (single-file)
- Streamlit UI
- Tiny built-in dataset (demo) to train a TF-IDF + LogisticRegression
- Lightweight rule-based checks for obvious scam signs
- Shows risk label, probability, and explanation (keywords + rules triggered)
"""

from pathlib import Path
import pickle
import re

import streamlit as st
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# -------------------------
# Demo dataset (small & labelled)
# -------------------------
DATA = [
    # ----- SCAM / PHISHING examples -----
    ("Your account has been suspended. Verify now at http://fake-bank-login.com", "scam"),
    ("URGENT: Update payment details or your subscription will be cancelled", "scam"),
    ("You've won a $1000 gift card! Click the link to claim", "scam"),
    ("Transfer 5000 INR immediately to avoid legal action", "scam"),
    ("Congrats! You are selected for a work-from-home job. Send bank details", "scam"),
    ("Verify your identity: send a photo of your ID to proceed", "scam"),
    ("This is the final notice! Pay penalty to avoid arrest", "scam"),
    ("Get a loan instantly with no credit check. Apply here", "scam"),
    ("We detected suspicious login. Click to reset password now", "scam"),
    ("Special offer: low price crypto investment, guaranteed returns", "scam"),
    ("Act now! Limited time investment opportunity. DM for details", "scam"),
    ("Congratulations! You've been chosen. Pay processing fee to receive prize", "scam"),
    ("Please confirm your OTP: 123456 — do not share this with anyone", "scam"),  # includes OTP prompt
    ("I need a small favor — send some money now, will repay tonight", "scam"),
    ("Click this tinyurl to view your invoice: tinyurl.com/fake", "scam"),
    ("Official-looking email asks for login details to avoid account lock", "scam"),
    ("Immediate transfer required. Use UPI id: pay@something", "scam"),
    ("You've been hired! Send payment for training materials", "scam"),
    ("Get refund — share card number and expiry to proceed", "scam"),
    ("Please review attached invoice and pay within 1 hour", "scam"),

    # ----- SAFE / NORMAL examples -----
    ("Hey, are we meeting at 6 pm today at the cafe?", "safe"),
    ("Reminder: project deadline is next Friday. Please push your code", "safe"),
    ("Can you share the lecture notes for last class? Thanks!", "safe"),
    ("Happy birthday! Hope you have an amazing day 🎉", "safe"),
    ("Your order has been shipped. Track here on the official site.", "safe"),
    ("Let's study together for the midterm tomorrow — are you free?", "safe"),
    ("I sent the assignment PDF to your email, check and confirm", "safe"),
    ("Dinner at my place tonight? I made paneer—let me know if you come", "safe"),
    ("Please review my PR and leave comments if any changes needed", "safe"),
    ("Congrats on your internship offer! Well deserved.", "safe"),
    ("Can you forward the meeting recording? I missed it.", "safe"),
    ("Team: standup at 10:00, please be on time. Agenda below.", "safe"),
    ("Thanks for your help on the lab — I learned a lot.", "safe"),
    ("Would you like to join the study group this Sunday?", "safe"),
    ("Parking is full today — use the side gate for entry", "safe"),
    ("Invoice from our vendor attached — please verify amounts", "safe"),
    ("I will be on leave tomorrow, please assign urgent tickets to me today", "safe"),
    ("Sharing code snippet that fixed the bug in function X", "safe"),
    ("Please review the syllabus changes posted by the faculty", "safe"),
    ("Movie night plan: let's vote for a movie by Friday", "safe"),
]

# -------------------------
# Simple rule-based scam indicators
# -------------------------
SCAM_KEYWORDS = [
    "verify", "verify now", "verify your", "suspended", "suspend", "urgent",
    "immediately", "transfer", "click", "claim", "won", "winner", "congrat",
    "pay", "payment", "bank", "loan", "credit", "otp", "processing fee",
    "limited time", "guaranteed returns", "tinyurl", "bit.ly", "account has been",
    "send bank", "send money", "legal action", "final notice", "selected for",
    "work-from-home", "training materials", "pay penalty"
]

URL_REGEX = re.compile(r"https?://|www\.|tinyurl\.|bit\.ly", re.IGNORECASE)
UPI_REGEX = re.compile(r"@\w+$")  # simplified UPI handle pattern

MODEL_PATH = Path("scam_detector.pkl")

# -------------------------
# Training / Loading model
# -------------------------
def train_and_save_model(data=DATA, model_path=MODEL_PATH):
    texts = [t for t, _ in data]
    y = np.array([1 if label == "scam" else 0 for _, label in data])

    vec = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
    clf = LogisticRegression(max_iter=1000)

    pipe = make_pipeline(vec, clf)
    pipe.fit(texts, y)

    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)

    return pipe

def load_model(model_path=MODEL_PATH):
    if model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        return train_and_save_model()

# -------------------------
# Explanation helpers
# -------------------------
def find_keywords(text):
    found = []
    lower = text.lower()
    for kw in SCAM_KEYWORDS:
        if kw in lower:
            found.append(kw)
    if URL_REGEX.search(text):
        found.append("url/link")
    if UPI_REGEX.search(text):
        found.append("upi-handle")
    # OTP pattern: 4-6 digit number and 'otp' or 'code'
    if re.search(r"\botp\b", lower) or re.search(r"\b(code|passcode)\b", lower):
        found.append("otp/code mention")
    if re.search(r"\b\d{4,6}\b", text):
        # could be OTP-like; only add if 'otp' not already included
        if "otp/code mention" not in found:
            found.append("number (possible OTP)")
    return found

def rule_based_score(text):
    score = 0
    lower = text.lower()
    # If it contains explicit money/payment/urgent words -> bump score
    if any(word in lower for word in ["transfer", "pay", "payment", "send money", "bank", "penalty", "processing fee"]):
        score += 0.35
    if "urgent" in lower or "immediately" in lower or "final notice" in lower:
        score += 0.25
    if URL_REGEX.search(text):
        score += 0.2
    if "verify" in lower and ("account" in lower or "identity" in lower or "login" in lower):
        score += 0.2
    # cap at 1.0
    return min(score, 1.0)

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Smart Cyber-Safety Analyzer", page_icon="🛡️")
st.title("🛡️ Smart Cyber-Safety Analyzer (Demo)")
st.write("Paste a message (email/DM/whatsapp text). The tool gives a risk score, explains why, and shows the probability from a tiny ML model.")

# Load or train model (quick for this tiny dataset)
with st.spinner("Loading model..."):
    model = load_model()

text_input = st.text_area("Message text", height=160, value="Hi, your account has been suspended. Verify now at http://fake-bank-login.com")

if st.button("Analyze"):
    if not text_input.strip():
        st.warning("Enter some text to analyze.")
    else:
        # ML prediction
        prob_scam = float(model.predict_proba([text_input])[0][1])  # probability of being scam
        ml_score = prob_scam  # 0..1

        # rule-based score
        rb = rule_based_score(text_input)

        # Combined heuristic: weighted average (simple)
        combined = min(1.0, 0.6 * ml_score + 0.4 * rb)

        # Labeling thresholds (tweakable)
        if combined >= 0.66:
            label = "High Risk (Likely Scam)"
            badge = ":red_circle:"
        elif combined >= 0.33:
            label = "Medium Risk (Suspicious)"
            badge = ":orange_circle:"
        else:
            label = "Low Risk (Likely Safe)"
            badge = ":green_circle:"

        st.markdown(f"### {badge} {label}")
        st.write(f"**Risk score:** {combined:.2f}  (ML: {ml_score:.2f} | Rules: {rb:.2f})")

        # Explanations
        st.markdown("**Why:**")
        reasons = []
        kws = find_keywords(text_input)
        if kws:
            reasons.append("Detected suspicious keywords / patterns: " + ", ".join(kws))
        # show top model features contributing? (quick approx)
        try:
            # get vectorizer and classifier from pipeline
            vec = model.named_steps['tfidfvectorizer']
            clf = model.named_steps['logisticregression']
            x = vec.transform([text_input])
            # compute coef * x to find important features present
            if hasattr(clf, "coef_"):
                coefs = clf.coef_[0]
                # map feature names to coef
                feature_names = np.array(vec.get_feature_names_out())
                # Get non-zero features in x
                nz = x.nonzero()[1]
                if len(nz) > 0:
                    feat_scores = list(zip(feature_names[nz], coefs[nz]))
                    feat_scores = sorted(feat_scores, key=lambda t: -abs(t[1]))[:6]
                    feat_msgs = [f"{f} ({s:.2f})" for f, s in feat_scores]
                    reasons.append("Model-relevant tokens: " + ", ".join(feat_msgs))
        except Exception:
            pass

        if not reasons:
            reasons.append("No obvious scam indicators detected by rules or model.")
        for r in reasons:
            st.write("- " + r)

        # Quick suggestions
        st.markdown("**Quick suggestions:**")
        suggestions = []
        if combined >= 0.66:
            suggestions.append("Do not click any links or share personal/banking details.")
            suggestions.append("Verify the sender via a known channel (official website/contact).")
            suggestions.append("Report or block the sender if it's a DM/email.")
        elif combined >= 0.33:
            suggestions.append("Be cautious: double-check links and requests for money or OTPs.")
            suggestions.append("Ask the sender for details via a trusted channel before acting.")
        else:
            suggestions.append("Looks safe, but always avoid sharing OTPs, passwords, or bank details.")
        for s in suggestions:
            st.write("- " + s)

# Footer: small demo dataset / run info
st.markdown("---")
st.caption("Demo model trained on a tiny in-app dataset (for showcase). For production, replace dataset with a larger labeled corpus, add cross-validation, and deploy securely.")
