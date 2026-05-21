# ================================================================
#  FAKE JOB / INTERNSHIP SCAM DETECTOR
#  PHASE 2 — ML Model Training
#
#  Input  : fake_jobs_cleaned.csv  (Phase 1 se)
#  Output : scam_detector_model.pkl (Streamlit app mein use hoga)
#
#  Algorithm : Random Forest + TF-IDF
#  Target    : fraudulent (0=Real, 1=Fake)
# ================================================================


# ──────────────────────────────────────────────
#  STEP 1 : Libraries
# ──────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection      import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble             import RandomForestClassifier
from sklearn.linear_model         import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing        import StandardScaler
from sklearn.metrics              import (classification_report, confusion_matrix,
                                          f1_score, roc_auc_score, roc_curve)
from sklearn.pipeline             import Pipeline
from scipy.sparse                 import hstack, csr_matrix
from imblearn.over_sampling       import SMOTE

# Agar imblearn nahi hai toh install karo:
# !pip install imbalanced-learn -q

FAKE_COLOR = '#E53935'
REAL_COLOR = '#1E88E5'
print("✅ STEP 1 Done — Libraries load ho gayi!")


# ──────────────────────────────────────────────
#  STEP 2 : Data load karo
# ──────────────────────────────────────────────
df = pd.read_csv('fake_jobs_cleaned.csv')

# Text columns ko safely fill karo
text_cols = ['title', 'company_profile', 'description',
             'requirements', 'benefits', 'combined_text']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].fillna('')

print(f"✅ STEP 2 Done — Data loaded!")
print(f"   Shape  : {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"   Fake   : {df['fraudulent'].sum():,}  ({df['fraudulent'].mean()*100:.1f}%)")
print(f"   Real   : {(df['fraudulent']==0).sum():,}  ({(df['fraudulent']==0).mean()*100:.1f}%)")


# ──────────────────────────────────────────────
#  STEP 3 : Features banao
#
#  Do type ke features:
#  A) Numeric  — binary flags + text lengths
#  B) Text     — combined_text pe TF-IDF
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3 — Feature Engineering")
print("="*60)

# A) Numeric features
numeric_features = ['telecommuting', 'has_company_logo', 'has_questions']

# Text length features (agar column hai toh)
for col in ['description', 'company_profile', 'requirements']:
    len_col = col + '_len'
    if len_col not in df.columns:
        df[len_col] = df[col].apply(len)
    numeric_features.append(len_col)

# Extra engineered features
df['has_salary']       = df['salary_range'].fillna('').apply(lambda x: 1 if len(x) > 0 else 0) \
                         if 'salary_range' in df.columns else 0
df['profile_missing']  = (df['company_profile'].apply(len) == 0).astype(int)
df['req_missing']      = (df['requirements'].apply(len) == 0).astype(int)
df['combined_len']     = df['combined_text'].apply(len)

numeric_features += ['profile_missing', 'req_missing', 'combined_len']

print(f"\n   Numeric features ({len(numeric_features)}):")
for f in numeric_features:
    print(f"   - {f}")

# B) TF-IDF on combined_text
print(f"\n   Text feature: combined_text → TF-IDF")

X_text_raw = df['combined_text']
X_num_raw  = df[numeric_features].fillna(0)
y          = df['fraudulent']

print(f"\n   Features ready!")
print(f"   Numeric : {X_num_raw.shape[1]} columns")
print(f"   Text    : TF-IDF banayenge")


# ──────────────────────────────────────────────
#  STEP 4 : Train/Test split
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4 — Train/Test Split (80/20)")
print("="*60)

X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = \
    train_test_split(X_text_raw, X_num_raw, y,
                     test_size=0.20,
                     random_state=42,
                     stratify=y)   # stratify = fake/real ratio maintain karo

print(f"\n   Train set : {len(y_train):,} rows  (Fake: {y_train.sum():,})")
print(f"   Test set  : {len(y_test):,}  rows  (Fake: {y_test.sum():,})")


# ──────────────────────────────────────────────
#  STEP 5 : TF-IDF Vectorizer
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5 — TF-IDF Vectorization")
print("="*60)

tfidf = TfidfVectorizer(
    max_features   = 5000,    # Top 5000 words rakhenge
    ngram_range    = (1, 2),  # Single words + 2-word phrases
    min_df         = 3,       # Kam se kam 3 docs mein aana chahiye
    max_df         = 0.95,    # Bahut common words ignore
    sublinear_tf   = True,    # Log scaling
    strip_accents  = 'unicode',
    analyzer       = 'word',
    stop_words     = 'english'
)

X_tfidf_train = tfidf.fit_transform(X_text_train)
X_tfidf_test  = tfidf.transform(X_text_test)

print(f"   TF-IDF matrix shape (train) : {X_tfidf_train.shape}")
print(f"   TF-IDF matrix shape (test)  : {X_tfidf_test.shape}")
print(f"   Vocabulary size             : {len(tfidf.vocabulary_):,} words")

# Numeric features ko sparse matrix mein convert karo
X_num_train_sp = csr_matrix(X_num_train.values)
X_num_test_sp  = csr_matrix(X_num_test.values)

# Combine — TF-IDF + Numeric
X_train_combined = hstack([X_tfidf_train, X_num_train_sp])
X_test_combined  = hstack([X_tfidf_test,  X_num_test_sp])

print(f"\n   Final feature matrix (train) : {X_train_combined.shape}")
print(f"   Final feature matrix (test)  : {X_test_combined.shape}")


# ──────────────────────────────────────────────
#  STEP 6 : SMOTE — Imbalanced data handle karo
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6 — SMOTE (Imbalanced Data Fix)")
print("="*60)

print(f"\n   Before SMOTE:")
print(f"   Real : {(y_train==0).sum():,}  |  Fake : {(y_train==1).sum():,}")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train_combined, y_train)

print(f"\n   After SMOTE:")
print(f"   Real : {(y_train_sm==0).sum():,}  |  Fake : {(y_train_sm==1).sum():,}")
print(f"   ✅ Ab balanced dataset se model train hoga!")


# ──────────────────────────────────────────────
#  STEP 7 : Model Training — Random Forest
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 7 — Random Forest Training  (2-3 min lagenge)")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators     = 200,         # 200 trees
    max_depth        = 20,          # Zyada deep nahi jaayenge
    min_samples_leaf = 2,
    class_weight     = 'balanced',  # Extra precaution for imbalance
    random_state     = 42,
    n_jobs           = -1           # Saare CPU cores use karo
)

print("\n   Training ho raha hai... thoda wait karo ⏳")
rf_model.fit(X_train_sm, y_train_sm)
print("   ✅ Model train ho gaya!")


# ──────────────────────────────────────────────
#  STEP 8 : Model Evaluation
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 8 — Model Evaluation")
print("="*60)

y_pred      = rf_model.predict(X_test_combined)
y_pred_prob = rf_model.predict_proba(X_test_combined)[:, 1]

f1  = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"\n   📊 Results:")
print(f"   F1 Score  : {f1:.4f}  (1.0 = perfect)")
print(f"   ROC-AUC   : {auc:.4f}  (1.0 = perfect)")
print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

# ── Plot 1 : Confusion Matrix ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'],
            linewidths=1, linecolor='white',
            annot_kws={'size': 14, 'weight': 'bold'})
axes[0].set_title('Confusion Matrix', fontweight='bold', fontsize=13)
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# True Negative, False Positive, False Negative, True Positive
tn, fp, fn, tp = cm.ravel()
print(f"\n   Confusion Matrix breakdown:")
print(f"   ✅ Sahi pakde Real jobs (TN) : {tn:,}")
print(f"   ✅ Sahi pakde Fake jobs (TP) : {tp:,}")
print(f"   ❌ Real ko Fake bola (FP)    : {fp:,}")
print(f"   ❌ Fake ko Real bola (FN)    : {fn:,}")

# ── Plot 2 : ROC Curve ──
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1].plot(fpr, tpr, color=FAKE_COLOR, lw=2,
             label=f'Random Forest (AUC = {auc:.3f})')
axes[1].plot([0,1], [0,1], 'k--', lw=1, label='Random baseline')
axes[1].fill_between(fpr, tpr, alpha=0.1, color=FAKE_COLOR)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve', fontweight='bold', fontsize=13)
axes[1].legend(loc='lower right')

plt.suptitle('Model Evaluation — Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot8_model_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Plot 8 save hua → plot8_model_evaluation.png")


# ──────────────────────────────────────────────
#  STEP 9 : Feature Importance — Kaun se features
#           model ke liye sabse important hain?
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 9 — Feature Importance")
print("="*60)

# Feature names banao
tfidf_feature_names = tfidf.get_feature_names_out().tolist()
all_feature_names   = tfidf_feature_names + numeric_features

importances = rf_model.feature_importances_
feat_df = pd.DataFrame({
    'feature'    : all_feature_names,
    'importance' : importances
}).sort_values('importance', ascending=False)

# Top 20 features
top20 = feat_df.head(20)
print(f"\n   Top 20 most important features:")
for i, (_, row) in enumerate(top20.iterrows(), 1):
    print(f"   {i:>2}. {row['feature']:<30} {row['importance']:.4f}")

# Separate: numeric vs text features ki importance
num_imp  = feat_df[feat_df['feature'].isin(numeric_features)].copy()
text_imp = feat_df[~feat_df['feature'].isin(numeric_features)].head(15)

# Plot 9 : Top features
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Text features
text_imp_sorted = text_imp.sort_values('importance')
axes[0].barh(text_imp_sorted['feature'], text_imp_sorted['importance'],
             color=REAL_COLOR, edgecolor='white')
axes[0].set_title('Top 15 TF-IDF Words\n(Text features)', fontweight='bold')
axes[0].set_xlabel('Feature Importance')

# Numeric features
num_imp_sorted = num_imp.sort_values('importance')
clrs = [FAKE_COLOR if f in ['has_company_logo', 'profile_missing'] else
        '#FFA726'  if f in ['has_questions', 'req_missing'] else
        REAL_COLOR for f in num_imp_sorted['feature']]
axes[1].barh(num_imp_sorted['feature'], num_imp_sorted['importance'],
             color=clrs, edgecolor='white')
axes[1].set_title('Numeric Features Importance\n(Laal = Key Red Flags)', fontweight='bold')
axes[1].set_xlabel('Feature Importance')

plt.suptitle('Feature Importance — Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot9_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 9 save hua → plot9_feature_importance.png")


# ──────────────────────────────────────────────
#  STEP 10 : Cross Validation — Model reliable hai?
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 10 — Cross Validation (5-Fold)")
print("="*60)

print("\n   Running 5-fold CV... (thoda time lagega)")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_train_combined, y_train,
                             cv=cv, scoring='f1', n_jobs=-1)

print(f"\n   CV F1 Scores : {[f'{s:.3f}' for s in cv_scores]}")
print(f"   Mean F1      : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

if cv_scores.std() < 0.05:
    print(f"   ✅ Model stable hai — consistent results aa rahe hain!")
else:
    print(f"   ⚠️  Thoda variance hai — more data se improve hoga")


# ──────────────────────────────────────────────
#  STEP 11 : Model save karo (.pkl file)
#  Streamlit app mein yahi load hoga
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 11 — Model Save karo")
print("="*60)

# Saari cheezein ek dict mein pack karo
model_package = {
    'model'            : rf_model,
    'tfidf'            : tfidf,
    'numeric_features' : numeric_features,
    'feature_names'    : all_feature_names,
    'model_info'       : {
        'f1_score'  : round(f1, 4),
        'roc_auc'   : round(auc, 4),
        'train_rows': len(y_train),
        'test_rows' : len(y_test),
    }
}

with open('scam_detector_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print(f"\n   ✅ scam_detector_model.pkl save ho gaya!")
print(f"   📦 Model package contains:")
print(f"      - Random Forest model  (200 trees)")
print(f"      - TF-IDF vectorizer    (5000 features)")
print(f"      - Numeric features list")
print(f"      - Model performance info")


# ──────────────────────────────────────────────
#  STEP 12 : Live test — Ek fake job predict karo!
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 12 — Live Test! (Real example pe predict karo)")
print("="*60)

def predict_job(title, description, company_profile='',
                requirements='', benefits='',
                has_logo=1, has_questions=1, telecommuting=0):
    """Ek job posting ko analyze karo"""

    combined = f"{title} {company_profile} {description} {requirements} {benefits}"

    # TF-IDF
    text_vec = tfidf.transform([combined])

    # Numeric
    num_data = {
        'telecommuting'       : telecommuting,
        'has_company_logo'    : has_logo,
        'has_questions'       : has_questions,
        'description_len'     : len(description),
        'company_profile_len' : len(company_profile),
        'requirements_len'    : len(requirements),
        'profile_missing'     : 1 if len(company_profile) == 0 else 0,
        'req_missing'         : 1 if len(requirements)    == 0 else 0,
        'combined_len'        : len(combined),
    }
    num_vec = csr_matrix([[num_data.get(f, 0) for f in numeric_features]])
    combined_vec = hstack([text_vec, num_vec])

    prob  = rf_model.predict_proba(combined_vec)[0][1]
    pred  = rf_model.predict(combined_vec)[0]
    score = int(prob * 100)

    verdict = "🚨 SCAM!" if score > 70 else "⚠️ Suspicious" if score > 40 else "✅ Legit"
    print(f"\n   Job Title   : {title}")
    print(f"   Risk Score  : {score}/100")
    print(f"   Verdict     : {verdict}")
    return score, pred

# Test 1 — Obvious scam
print("\n   TEST 1 — Obvious Scam Job:")
predict_job(
    title          = "Data Entry Work From Home Earn Daily",
    description    = "Easy work, no experience needed. Earn Rs 50,000 per month. Send Aadhaar card.",
    company_profile= "",
    requirements   = "",
    has_logo       = 0,
    has_questions  = 0,
    telecommuting  = 1
)

# Test 2 — Legitimate job
print("\n   TEST 2 — Legitimate Job:")
predict_job(
    title          = "Software Engineer - Backend",
    description    = "We are looking for an experienced backend engineer to join our team. "
                     "You will work on scalable microservices using Python and Django. "
                     "5+ years experience required. Competitive salary and benefits.",
    company_profile= "TechCorp is a leading software company founded in 2010 with 500+ employees.",
    requirements   = "5+ years Python experience, knowledge of REST APIs, SQL databases, "
                     "excellent communication skills, B.Tech/MCA degree required.",
    has_logo       = 1,
    has_questions  = 1,
    telecommuting  = 0
)

# Test 3 — Internship scam (tumhara use case!)
print("\n   TEST 3 — Fake Internship:")
predict_job(
    title          = "Home Based Online Internship Part Time",
    description    = "Work from home internship. No experience needed. "
                     "Earn 20,000 per month. Registration fee Rs 500.",
    company_profile= "",
    requirements   = "",
    has_logo       = 0,
    has_questions  = 0,
    telecommuting  = 1
)


# ──────────────────────────────────────────────
#  FINAL SUMMARY
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("🎉 PHASE 2 COMPLETE!")
print("="*60)
print(f"""
╔══════════════════════════════════════════════════════════╗
║         MODEL PERFORMANCE SUMMARY                        ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Algorithm  : Random Forest (200 trees)                  ║
║  F1 Score   : {f1:.4f}                                      ║
║  ROC-AUC    : {auc:.4f}                                      ║
║                                                          ║
║  Features used:                                          ║
║  - TF-IDF (5000 words from job text)                     ║
║  - Binary flags (logo, questions, telecommuting)         ║
║  - Text length features                                  ║
║  - SMOTE for imbalanced data                             ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║  Files saved:                                            ║
║  - scam_detector_model.pkl  ← Streamlit app use karega   ║
║  - plot8_model_evaluation.png                            ║
║  - plot9_feature_importance.png                          ║
╠══════════════════════════════════════════════════════════╣
║  NEXT → Phase 3: Streamlit App banana!                   ║
╚══════════════════════════════════════════════════════════╝
""")
