# ================================================================
#  FAKE JOB / INTERNSHIP SCAM DETECTOR
#  PHASE 3 — Streamlit Web App
#
#  Run locally  : streamlit run app.py
#  Deploy       : Streamlit Cloud (free) — github se connect karo
#
#  Required files in same folder:
#  - app.py                    (this file)
#  - scam_detector_model.pkl   (from Phase 2)
#  - requirements.txt          (dependencies)
# ================================================================

import streamlit as st
import pickle
import numpy as np
from scipy.sparse import hstack, csr_matrix

# ──────────────────────────────────────────────
#  Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title = "Job Scam Detector",
    page_icon  = "🛡️",
    layout     = "centered"
)

# ──────────────────────────────────────────────
#  Custom CSS Styling
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .score-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .score-safe {
        background: #e8f5e9;
        border: 2px solid #4caf50;
    }
    .score-suspicious {
        background: #fff3e0;
        border: 2px solid #ff9800;
    }
    .score-scam {
        background: #ffebee;
        border: 2px solid #f44336;
    }
    .score-number {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
    }
    .verdict-text {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    .flag-item {
        padding: 0.5rem 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        font-size: 0.95rem;
    }
    .flag-high   { background: #ffebee; border-left: 4px solid #f44336; }
    .flag-medium { background: #fff3e0; border-left: 4px solid #ff9800; }
    .flag-low    { background: #e8f5e9; border-left: 4px solid #4caf50; }
    .tip-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  Load Model (cached so it loads only once)
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('scam_detector_model.pkl', 'rb') as f:
        package = pickle.load(f)
    return package

try:
    pkg              = load_model()
    model            = pkg['model']
    tfidf            = pkg['tfidf']
    numeric_features = pkg['numeric_features']
    model_info       = pkg.get('model_info', {})
    MODEL_LOADED     = True
except FileNotFoundError:
    MODEL_LOADED = False


# ──────────────────────────────────────────────
#  Prediction Function
# ──────────────────────────────────────────────
def predict(title, description, company_profile,
            requirements, benefits,
            has_logo, has_questions, telecommuting):
    """
    Run the ML model and return:
    - risk_score (0-100)
    - verdict    (LEGIT / SUSPICIOUS / SCAM)
    - red_flags  (list of detected issues)
    - green_flags(list of positive signals)
    """
    combined = f"{title} {company_profile} {description} {requirements} {benefits}"

    # TF-IDF transform
    text_vec = tfidf.transform([combined])

    # Numeric features
    num_vals = {
        'telecommuting'      : int(telecommuting),
        'has_company_logo'   : int(has_logo),
        'has_questions'      : int(has_questions),
        'description_len'    : len(description),
        'company_profile_len': len(company_profile),
        'requirements_len'   : len(requirements),
        'profile_missing'    : 1 if len(company_profile.strip()) == 0 else 0,
        'req_missing'        : 1 if len(requirements.strip())    == 0 else 0,
        'combined_len'       : len(combined),
    }
    num_vec      = csr_matrix([[num_vals.get(f, 0) for f in numeric_features]])
    combined_vec = hstack([text_vec, num_vec])

    # Predict
    prob  = model.predict_proba(combined_vec)[0][1]
    score = int(prob * 100)

    # Assign verdict
    if score > 70:
        verdict = "SCAM"
    elif score > 40:
        verdict = "SUSPICIOUS"
    else:
        verdict = "LEGIT"

    # Rule-based red flag detection (for user explanation)
    red_flags   = []
    green_flags = []

    # High severity flags
    if not has_logo:
        red_flags.append(("HIGH", "No company logo found — unverified company"))
    if not has_questions:
        red_flags.append(("HIGH", "No screening questions — no proper hiring process"))
    if len(company_profile.strip()) < 50:
        red_flags.append(("HIGH", "Company profile missing or very short"))
    if len(requirements.strip()) < 30:
        red_flags.append(("HIGH", "No proper requirements listed"))

    # Keyword-based flags
    text_lower = combined.lower()
    scam_keywords = {
        "registration fee"  : "Asking for registration fee — big red flag!",
        "training fee"      : "Asking for training fee — scammers do this",
        "send aadhaar"      : "Asking for Aadhaar card upfront — never share this",
        "send pan"          : "Asking for PAN card upfront — suspicious",
        "bank details"      : "Asking for bank details — very suspicious",
        "earn daily"        : "'Earn daily' claim — unrealistic promise",
        "no experience needed": "No experience for high salary — too good to be true",
        "earn lakhs"        : "Claiming to earn lakhs — likely fake",
        "whatsapp only"     : "WhatsApp-only contact — unprofessional",
        "data entry"        : "Data entry jobs have very high scam rate",
        "home based"        : "Home-based job claims need careful verification",
        "work from home"    : "Work-from-home needs extra verification",
        "earn money online" : "Vague 'earn money online' claim",
        "form filling"      : "Form filling jobs are commonly fake",
        "part time online"  : "Part-time online jobs are frequently scams",
    }
    for keyword, message in scam_keywords.items():
        if keyword in text_lower:
            red_flags.append(("MEDIUM", f"Keyword detected: '{keyword}' — {message}"))

    # Salary check
    if len(description) > 20:
        import re
        salary_matches = re.findall(r'(\d[\d,]*)\s*(?:per month|/month|pm)', text_lower)
        for s in salary_matches:
            amount = int(s.replace(',', ''))
            if amount > 40000:
                red_flags.append(("MEDIUM",
                    f"Very high salary claim (Rs {s}/month) for unclear role — verify carefully"))

    if telecommuting:
        red_flags.append(("LOW", "Remote/telecommuting job — verify the company independently"))

    # Green flags
    if has_logo:
        green_flags.append("Company logo present — adds credibility")
    if has_questions:
        green_flags.append("Screening questions present — proper hiring process")
    if len(company_profile.strip()) > 200:
        green_flags.append("Detailed company profile provided")
    if len(requirements.strip()) > 100:
        green_flags.append("Clear requirements listed")
    if len(description.strip()) > 500:
        green_flags.append("Detailed job description — sign of a real posting")

    return score, verdict, red_flags, green_flags


# ──────────────────────────────────────────────
#  MAIN APP UI
# ──────────────────────────────────────────────

# Header
st.markdown('<div class="main-title">🛡️ Job Scam Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Paste any job or internship posting — AI will detect if it is fake</div>',
            unsafe_allow_html=True)

# Model not found warning
if not MODEL_LOADED:
    st.error("⚠️ Model file not found! Make sure `scam_detector_model.pkl` is in the same folder as `app.py`")
    st.info("Run Phase 2 first to generate the model file.")
    st.stop()

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(["🔍 Analyze Job", "📊 Model Info", "ℹ️ How It Works"])

# ────────────────────────────────────────────────────────
#  TAB 1 — Main Analysis
# ────────────────────────────────────────────────────────
with tab1:

    st.subheader("Enter Job Posting Details")

    # Row 1 — Title
    title = st.text_input(
        "Job Title *",
        placeholder="e.g. Data Entry Operator, Software Engineer Intern"
    )

    # Row 2 — Description
    description = st.text_area(
        "Job Description *",
        placeholder="Paste the full job description here...",
        height=160
    )

    # Row 3 — Company profile + Requirements side by side
    col1, col2 = st.columns(2)
    with col1:
        company_profile = st.text_area(
            "Company Profile",
            placeholder="About the company...",
            height=120
        )
    with col2:
        requirements = st.text_area(
            "Requirements",
            placeholder="Skills, experience, education needed...",
            height=120
        )

    # Row 4 — Benefits
    benefits = st.text_area(
        "Benefits / Perks (optional)",
        placeholder="Salary, health insurance, work hours...",
        height=80
    )

    # Row 5 — Binary flags
    st.subheader("Job Posting Flags")
    col3, col4, col5 = st.columns(3)
    with col3:
        has_logo      = st.checkbox("Company Logo Present", value=True)
    with col4:
        has_questions = st.checkbox("Screening Questions Present", value=True)
    with col5:
        telecommuting = st.checkbox("Remote / Work From Home", value=False)

    st.divider()

    # Analyze button
    analyze_btn = st.button("🔍 Analyze This Job", type="primary", use_container_width=True)

    if analyze_btn:
        # Validate required fields
        if not title.strip() and not description.strip():
            st.warning("Please enter at least a Job Title or Description.")
        else:
            with st.spinner("Analyzing... please wait"):
                score, verdict, red_flags, green_flags = predict(
                    title, description, company_profile,
                    requirements, benefits,
                    has_logo, has_questions, telecommuting
                )

            # ── Result Box ──
            if verdict == "SCAM":
                css_class  = "score-scam"
                icon       = "🚨"
                verdict_msg= "HIGH RISK — DO NOT APPLY"
                color      = "#f44336"
            elif verdict == "SUSPICIOUS":
                css_class  = "score-suspicious"
                icon       = "⚠️"
                verdict_msg= "SUSPICIOUS — Verify Before Applying"
                color      = "#ff9800"
            else:
                css_class  = "score-safe"
                icon       = "✅"
                verdict_msg= "Looks Legitimate"
                color      = "#4caf50"

            st.markdown(f"""
            <div class="score-box {css_class}">
                <div class="score-number" style="color:{color}">{score}</div>
                <div style="color:#888; font-size:0.9rem">Risk Score (out of 100)</div>
                <div class="verdict-text" style="color:{color}">{icon} {verdict_msg}</div>
            </div>
            """, unsafe_allow_html=True)

            # Risk meter using Streamlit progress bar
            st.markdown("**Risk Level:**")
            st.progress(score / 100)

            # ── Red Flags ──
            if red_flags:
                st.markdown("### 🚩 Red Flags Detected")
                for severity, message in red_flags:
                    css = "flag-high" if severity == "HIGH" else \
                          "flag-medium" if severity == "MEDIUM" else "flag-low"
                    badge = "🔴 HIGH" if severity == "HIGH" else \
                            "🟠 MEDIUM" if severity == "MEDIUM" else "🟢 LOW"
                    st.markdown(
                        f'<div class="flag-item {css}"><b>{badge}</b> — {message}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.success("No major red flags detected!")

            # ── Green Flags ──
            if green_flags:
                st.markdown("### ✅ Positive Signals")
                for gf in green_flags:
                    st.markdown(
                        f'<div class="flag-item flag-low">✅ {gf}</div>',
                        unsafe_allow_html=True
                    )

            # ── Safety Tips ──
            st.markdown("### 💡 Safety Tips")
            if verdict == "SCAM":
                tips = [
                    "Do NOT pay any registration or training fee.",
                    "Do NOT share Aadhaar, PAN, or bank details.",
                    "Report this job on the platform where you found it.",
                    "Block and ignore any further messages from this recruiter.",
                ]
            elif verdict == "SUSPICIOUS":
                tips = [
                    "Search the company name on Google before applying.",
                    "Check company reviews on Glassdoor or LinkedIn.",
                    "Never pay money to get a job — legitimate companies don't ask.",
                    "Call the company's official number to verify the job.",
                ]
            else:
                tips = [
                    "Still verify the company on LinkedIn before sharing personal info.",
                    "Never share bank details or ID proof before a formal offer letter.",
                    "Research the company's official website independently.",
                ]

            for tip in tips:
                st.markdown(
                    f'<div class="tip-box">💡 {tip}</div>',
                    unsafe_allow_html=True
                )


# ────────────────────────────────────────────────────────
#  TAB 2 — Model Info
# ────────────────────────────────────────────────────────
with tab2:
    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h2>98%</h2><p>Accuracy</p></div>',
                    unsafe_allow_html=True)
    with col2:
        f1_display = model_info.get('f1_score', 0.79)
        st.markdown(f'<div class="metric-card"><h2>{f1_display}</h2><p>F1 Score</p></div>',
                    unsafe_allow_html=True)
    with col3:
        auc_display = model_info.get('roc_auc', 0.98)
        st.markdown(f'<div class="metric-card"><h2>{auc_display}</h2><p>ROC-AUC</p></div>',
                    unsafe_allow_html=True)

    st.divider()
    st.subheader("Dataset Used")
    st.markdown("""
    - **Source**: [Real or Fake Job Postings — Kaggle](https://kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
    - **Total records**: 17,880 job postings
    - **Fake jobs**: 866 (4.8%)
    - **Real jobs**: 17,014 (95.2%)
    """)

    st.subheader("How the Model Works")
    st.markdown("""
    1. **TF-IDF Vectorizer** — converts job description text into 5000 numeric features
    2. **Numeric Features** — company logo, screening questions, text lengths, telecommuting flag
    3. **SMOTE** — balances the dataset (fake jobs were only 4.8%)
    4. **Random Forest** — 200 decision trees vote together to classify fake vs real
    """)

    st.subheader("Top Red Flag Features (from EDA)")
    flags_data = {
        "Feature"           : ["No company logo", "No company profile",
                               "No screening questions", "Short description",
                               "Keyword: data entry", "Keyword: work from home",
                               "Keyword: earn daily", "Part-time employment type"],
        "Scam Rate"         : ["15.9%", "High", "6.8%", "High",
                               "11.9% in fake titles", "3.1% in fake titles",
                               "High", "9.3%"],
        "Risk Level"        : ["HIGH", "HIGH", "HIGH", "MEDIUM",
                               "HIGH", "MEDIUM", "HIGH", "MEDIUM"]
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(flags_data), use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────────────
#  TAB 3 — How It Works
# ────────────────────────────────────────────────────────
with tab3:
    st.subheader("How to Use This Tool")
    st.markdown("""
    ### Step-by-step Guide

    1. **Copy** the job posting you received (from WhatsApp, LinkedIn, Internshala, email, etc.)
    2. **Paste** the job title and description into the form
    3. **Add** company profile and requirements if available
    4. **Check** the flags based on what you know about the posting
    5. **Click** Analyze — get instant results!

    ---

    ### Understanding the Risk Score

    | Score | Meaning | Action |
    |-------|---------|--------|
    | 0–40  | Likely Legitimate | Safe to apply, but always verify |
    | 41–70 | Suspicious | Research the company thoroughly first |
    | 71–100| High Risk Scam | Do NOT apply or share any information |

    ---

    ### Common Signs of a Fake Job / Internship

    - Asking for **registration fee** or **training fee**
    - Requesting **Aadhaar, PAN, or bank details** before hiring
    - **No company name** or vague company information
    - **Gmail / Yahoo** contact instead of official company email
    - **Unrealistic salary** — Rs 30,000–50,000/month for freshers WFH
    - **No experience required** for high-paying roles
    - **WhatsApp-only** communication
    - Pressure to **"apply within 24 hours"**
    - Jobs found via **random WhatsApp forwards**

    ---

    ### Frequently Asked Questions

    **Q: Is my data saved?**
    A: No. Nothing you type is stored anywhere. All analysis happens locally.

    **Q: Can it detect 100% of scams?**
    A: No tool is perfect. The model has 98% accuracy but always use your own judgment too.

    **Q: Where do most fake internships appear?**
    A: WhatsApp forwards, Telegram groups, and sometimes on smaller job boards.
    """)

    st.info("Built with Python, Scikit-learn, and Streamlit | Dataset: Kaggle (Shivam Bansal)")

