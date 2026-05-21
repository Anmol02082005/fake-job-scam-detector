# ================================================================
#  FAKE JOB / INTERNSHIP SCAM DETECTOR
#  PHASE 1 — Exploratory Data Analysis (EDA)
#
#  Dataset : Real or Fake Job Postings — Shivam Bansal (Kaggle)
#  Link    : kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
#  Rows    : 17,880  |  Columns : 18
#
#  Google Colab mein chalao — sab kuch step-by-step hai
# ================================================================


# ──────────────────────────────────────────────
#  STEP 0 : Dataset Kaggle se download karo
# ──────────────────────────────────────────────
# Google Colab mein in teen lines run karo (ek baar hi):
#
#   !pip install kaggle -q
#   from google.colab import files
#   files.upload()   # <-- apni kaggle.json API key upload karo
#
#   !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#   !kaggle datasets download -d shivamb/real-or-fake-fake-jobposting-prediction
#   !unzip -q real-or-fake-fake-jobposting-prediction.zip
#
# Ya seedha Kaggle se CSV download karke Colab mein upload karo:
#   from google.colab import files
#   uploaded = files.upload()   # fake_job_postings.csv select karo


# ──────────────────────────────────────────────
#  STEP 1 : Libraries import karo
# ──────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 60)

# Plot style
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size']      = 12
sns.set_style("whitegrid")
sns.set_palette("Set2")

FAKE_COLOR = '#E53935'   # laal = fake/scam
REAL_COLOR = '#1E88E5'   # neela = real/safe

print("✅ STEP 1 Done — Libraries load ho gayi!")


# ──────────────────────────────────────────────
#  STEP 2 : Dataset load karo
# ──────────────────────────────────────────────
CSV_PATH = "fake_job_postings.csv"   # <-- agar alag naam hai toh yahan badlo

df = pd.read_csv(CSV_PATH)

print(f"\n✅ STEP 2 Done — Dataset load hua!")
print(f"   📦 Total rows    : {df.shape[0]:,}")
print(f"   📋 Total columns : {df.shape[1]}")


# ──────────────────────────────────────────────
#  STEP 3 : Dataset ka overview
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3 — Dataset Overview")
print("="*60)

print("\n🔍 Sabhi 18 columns ki detail (null values ke saath):")
print(f"\n{'Column':<25} {'Dtype':<12} {'Null Count':>10} {'Null %':>8}")
print("-"*57)
for col in df.columns:
    nulls = df[col].isnull().sum()
    pct   = nulls / len(df) * 100
    print(f"{col:<25} {str(df[col].dtype):<12} {nulls:>10,} {pct:>7.1f}%")

print(f"\n🔍 Pehli 3 rows preview:")
print(df.head(3))

print(f"\n🔍 Columns ki list:")
print(list(df.columns))


# ──────────────────────────────────────────────
#  STEP 4 : Target variable — Fake vs Real
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4 — Fake vs Real (Target Variable: 'fraudulent')")
print("="*60)

counts     = df['fraudulent'].value_counts()
real_count = counts.get(0, 0)
fake_count = counts.get(1, 0)
total      = len(df)
fake_pct   = fake_count / total * 100
real_pct   = real_count / total * 100

print(f"\n   ✅ Real jobs (0) : {real_count:,}  ({real_pct:.1f}%)")
print(f"   🚨 Fake jobs (1) : {fake_count:,}  ({fake_pct:.1f}%)")
print(f"\n   ⚠️  Dataset bahut imbalanced hai — sirf {fake_pct:.1f}% fake hai")
print(f"   💡 Phase 2 mein class_weight='balanced' use karenge iske liye")

# Plot 1 : Fake vs Real
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Bar chart
bars = axes[0].bar(['Real Jobs', 'Fake Jobs'],
                   [real_count, fake_count],
                   color=[REAL_COLOR, FAKE_COLOR],
                   width=0.45, edgecolor='white', linewidth=1.5)
axes[0].set_title('Fake vs Real — Total Count', fontweight='bold', fontsize=13)
axes[0].set_ylabel('Number of Job Postings')
for bar in bars:
    h = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, h + 50,
                 f'{h:,}', ha='center', fontsize=12, fontweight='bold')
axes[0].set_ylim(0, real_count * 1.12)

# Pie chart
axes[1].pie([real_count, fake_count],
            labels=['Real Jobs', 'Fake Jobs'],
            colors=[REAL_COLOR, FAKE_COLOR],
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2},
            textprops={'fontsize': 12})
axes[1].set_title('Percentage Distribution', fontweight='bold', fontsize=13)

plt.suptitle("Target Variable: 'fraudulent'  (0=Real, 1=Fake)",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plot1_fake_vs_real.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Plot 1 save hua → plot1_fake_vs_real.png")


# ──────────────────────────────────────────────
#  STEP 5 : Missing Values ka analysis
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5 — Missing Values Analysis")
print("="*60)

null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
null_pct_nonzero = null_pct[null_pct > 0]

print(f"\n   {len(null_pct_nonzero)} columns mein missing values hain:\n")
for col, pct in null_pct_nonzero.items():
    filled = int(pct / 3)
    empty  = 33 - filled
    bar    = "█" * filled + "░" * empty
    print(f"   {col:<25}  {pct:>5.1f}%  [{bar}]")

print(f"\n   💡 Plan:")
print(f"      - salary_range, department → bahut zyada null, drop kar denge")
print(f"      - Baaki text columns → '' (empty string) se fill karenge")

# Plot 2 : Missing values bar chart
plt.figure(figsize=(11, 5))
colors = [FAKE_COLOR if p > 50 else '#FFA726' if p > 20 else REAL_COLOR
          for p in null_pct_nonzero.values]
bars = plt.bar(null_pct_nonzero.index, null_pct_nonzero.values,
               color=colors, edgecolor='white', linewidth=1)
plt.xticks(rotation=40, ha='right')
plt.ylabel('Missing Values (%)')
plt.title('Missing Values by Column\n(Laal = 50%+ missing, Orange = 20%+)',
          fontweight='bold')
plt.axhline(y=50, color=FAKE_COLOR, linestyle='--', alpha=0.5, label='50% line')
plt.axhline(y=20, color='orange',   linestyle='--', alpha=0.5, label='20% line')
plt.legend()
for bar in bars:
    h = bar.get_height()
    if h > 5:
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                 f'{h:.0f}%', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('plot2_missing_values.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 2 save hua → plot2_missing_values.png")


# ──────────────────────────────────────────────
#  STEP 6 : Binary Red Flag Features
#  (has_company_logo, has_questions, telecommuting)
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6 — Binary Features (Key Red Flags!)")
print("="*60)

binary_cols = ['telecommuting', 'has_company_logo', 'has_questions']
binary_cols = [c for c in binary_cols if c in df.columns]

print(f"\n   Binary feature mein 0 ya 1 hota hai:")
print(f"   - has_company_logo = 0 → company logo nahi hai → suspicious!")
print(f"   - has_questions = 0    → koi screening nahi → suspicious!")
print(f"   - telecommuting = 1    → work from home → check karo")

print(f"\n   Fake job rate for each value:")
print(f"   {'Feature':<20} {'Value':<8} {'Fake Rate':>10} {'Count':>8}")
print("   " + "-"*48)
for col in binary_cols:
    for val in sorted(df[col].dropna().unique()):
        subset    = df[df[col] == val]
        fake_rate = subset['fraudulent'].mean() * 100
        n         = len(subset)
        flag      = " ⬅️  RED FLAG!" if (col == 'has_company_logo' and val == 0) or \
                                         (col == 'has_questions'    and val == 0) else ""
        print(f"   {col:<20} {int(val):<8} {fake_rate:>9.1f}%  {n:>7,}{flag}")

# Plot 3 : Binary features fraud rate
fig, axes = plt.subplots(1, len(binary_cols), figsize=(14, 5))
for ax, col in zip(axes, binary_cols):
    grp  = df.groupby(col)['fraudulent'].mean() * 100
    vals = grp.index.astype(int).astype(str).tolist()
    clrs = []
    for v in grp.index:
        if (col == 'has_company_logo' and v == 0) or \
           (col == 'has_questions'    and v == 0):
            clrs.append(FAKE_COLOR)
        elif (col == 'telecommuting'  and v == 1):
            clrs.append('#FFA726')
        else:
            clrs.append(REAL_COLOR)
    bars = ax.bar(vals, grp.values, color=clrs, edgecolor='white',
                  linewidth=1.5, width=0.4)
    ax.set_title(col, fontweight='bold', fontsize=11)
    ax.set_xlabel('Value  (0=No, 1=Yes)')
    ax.set_ylabel('Fake Job Rate (%)')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                f'{h:.1f}%', ha='center', fontweight='bold', fontsize=12)

plt.suptitle('Binary Features — Fake Job Rate\n(Laal = High Risk Red Flag)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot3_binary_features.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 3 save hua → plot3_binary_features.png")


# ──────────────────────────────────────────────
#  STEP 7 : Employment Type vs Fraud Rate
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 7 — Employment Type vs Fraud Rate")
print("="*60)

if 'employment_type' in df.columns:
    emp = (df.groupby('employment_type')['fraudulent']
             .agg(['mean', 'count'])
             .reset_index())
    emp.columns  = ['employment_type', 'fake_rate', 'count']
    emp['fake_rate'] = emp['fake_rate'] * 100
    emp = emp[emp['count'] > 30].sort_values('fake_rate', ascending=True)
    avg = df['fraudulent'].mean() * 100

    print(f"\n   Employment Type → Fake Rate:")
    for _, row in emp.iterrows():
        flag = " 🚨" if row['fake_rate'] > avg * 1.5 else ""
        print(f"   {row['employment_type']:<20} {row['fake_rate']:>5.1f}%  "
              f"(n={row['count']:,}){flag}")

    plt.figure(figsize=(10, 5))
    clrs = [FAKE_COLOR if r > avg * 1.5 else
            '#FFA726'  if r > avg       else
            REAL_COLOR for r in emp['fake_rate']]
    plt.barh(emp['employment_type'], emp['fake_rate'],
             color=clrs, edgecolor='white', linewidth=1)
    plt.axvline(x=avg, color='gray', linestyle='--', linewidth=1.5,
                label=f'Average ({avg:.1f}%)')
    plt.xlabel('Fake Job Rate (%)')
    plt.title('Fraud Rate by Employment Type\n(Laal = High Risk)',
              fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot4_employment_type.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Plot 4 save hua → plot4_employment_type.png")


# ──────────────────────────────────────────────
#  STEP 8 : Required Experience vs Fraud Rate
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 8 — Required Experience vs Fraud Rate")
print("="*60)

if 'required_experience' in df.columns:
    exp = (df.groupby('required_experience')['fraudulent']
             .agg(['mean', 'count'])
             .reset_index())
    exp.columns  = ['experience', 'fake_rate', 'count']
    exp['fake_rate'] = exp['fake_rate'] * 100
    exp = exp[exp['count'] > 20].sort_values('fake_rate', ascending=False)
    avg = df['fraudulent'].mean() * 100

    print(f"\n   Experience level → Fake Rate:")
    for _, row in exp.iterrows():
        flag = " 🚨 MOST TARGETED!" if row['fake_rate'] == exp['fake_rate'].max() else ""
        print(f"   {row['experience']:<30} {row['fake_rate']:>5.1f}%  "
              f"(n={row['count']:,}){flag}")

    print(f"\n   💡 Entry level aur Internship positions pe scammers")
    print(f"      zyada target karte hain — fresh graduates ko dhyan rakhna chahiye!")


# ──────────────────────────────────────────────
#  STEP 9 : Text Length Analysis
#  Fake jobs mein description bahut chhoti hoti hai
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 9 — Text Length Analysis  (Important Feature!)")
print("="*60)

text_cols = ['description', 'company_profile', 'requirements', 'benefits']
text_cols = [c for c in text_cols if c in df.columns]

df_t = df.copy()
for col in text_cols:
    df_t[col + '_len'] = df_t[col].fillna('').apply(len)

print(f"\n   Average text length — Real vs Fake:")
print(f"   {'Column':<20} {'Real (avg chars)':>18} {'Fake (avg chars)':>18} {'Diff':>8}")
print("   " + "-"*66)
for col in text_cols:
    real_avg = df_t[df_t['fraudulent'] == 0][col + '_len'].mean()
    fake_avg = df_t[df_t['fraudulent'] == 1][col + '_len'].mean()
    diff     = real_avg - fake_avg
    print(f"   {col:<20} {real_avg:>18.0f} {fake_avg:>18.0f} {diff:>+8.0f}")

print(f"\n   💡 Real jobs mein description zyada detailed hoti hai")
print(f"      Fake jobs mein chhoti/vague description = RED FLAG!")

# Plot 5 : Text length distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, col in enumerate(text_cols):
    len_col   = col + '_len'
    real_data = df_t[df_t['fraudulent'] == 0][len_col].clip(upper=6000)
    fake_data = df_t[df_t['fraudulent'] == 1][len_col].clip(upper=6000)
    axes[i].hist(real_data, bins=60, alpha=0.65, color=REAL_COLOR,
                 label='Real', density=True)
    axes[i].hist(fake_data, bins=60, alpha=0.65, color=FAKE_COLOR,
                 label='Fake', density=True)
    axes[i].set_title(f"'{col}' length distribution", fontweight='bold')
    axes[i].set_xlabel('Character count  (clipped at 6000)')
    axes[i].set_ylabel('Density')
    axes[i].legend()

plt.suptitle('Text Length Distribution — Real vs Fake\n'
             'Fake jobs ki description chhoti hoti hai!',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot5_text_lengths.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 5 save hua → plot5_text_lengths.png")


# ──────────────────────────────────────────────
#  STEP 10 : Scam Keywords in Job Titles
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 10 — Scam Keywords in Job Titles")
print("="*60)

if 'title' in df.columns:
    fake_titles = df[df['fraudulent'] == 1]['title'].fillna('').str.lower()
    real_titles = df[df['fraudulent'] == 0]['title'].fillna('').str.lower()

    scam_keywords = [
        'data entry', 'home based', 'work from home', 'earn daily',
        'administrative', 'part time', 'part-time', 'online',
        'typing', 'form filling', 'freelance', 'no experience', 'earn money',
        'customer service', 'virtual assistant'
    ]

    kw_data = []
    print(f"\n   Keyword in title → Fake % vs Real %:")
    print(f"   {'Keyword':<22} {'Fake titles':>12} {'Real titles':>12}")
    print("   " + "-"*48)
    for kw in scam_keywords:
        f_pct = fake_titles.str.contains(kw, regex=False).mean() * 100
        r_pct = real_titles.str.contains(kw, regex=False).mean() * 100
        if f_pct > 0.5 or r_pct > 0.5:
            kw_data.append({'kw': kw, 'fake': f_pct, 'real': r_pct})
            flag = " 🚩" if f_pct > r_pct * 2 else ""
            print(f"   {kw:<22} {f_pct:>11.1f}%  {r_pct:>11.1f}%{flag}")

    if kw_data:
        import pandas as pd
        kdf = pd.DataFrame(kw_data).sort_values('fake')
        x   = range(len(kdf))
        w   = 0.35
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh([i - w/2 for i in x], kdf['fake'], w,
                color=FAKE_COLOR, label='Fake jobs', alpha=0.85)
        ax.barh([i + w/2 for i in x], kdf['real'], w,
                color=REAL_COLOR, label='Real jobs', alpha=0.85)
        ax.set_yticks(list(x))
        ax.set_yticklabels(kdf['kw'], fontsize=11)
        ax.set_xlabel('Occurrence in Job Titles (%)')
        ax.set_title('Scam Keywords in Job Titles\n'
                     '(Fake mein zyada → suspicious)',
                     fontweight='bold')
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig('plot6_scam_keywords.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✅ Plot 6 save hua → plot6_scam_keywords.png")


# ──────────────────────────────────────────────
#  STEP 11 : Correlation Heatmap (numeric cols)
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 11 — Correlation Heatmap")
print("="*60)

num_cols = ['telecommuting', 'has_company_logo', 'has_questions', 'fraudulent']
num_cols = [c for c in num_cols if c in df.columns]

plt.figure(figsize=(7, 5))
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, mask=mask,
            linewidths=0.5, annot_kws={'size': 13})
plt.title('Correlation — Numeric Features vs Fraudulent',
          fontweight='bold')
plt.tight_layout()
plt.savefig('plot7_correlation.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot 7 save hua → plot7_correlation.png")


# ──────────────────────────────────────────────
#  STEP 12 : Clean Dataset — Phase 2 ke liye save karo
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 12 — Cleaned Dataset Save karo (Phase 2 ke liye)")
print("="*60)

df_clean = df.copy()

# Drop karo — bahut zyada null hain inme
drop_cols = ['salary_range', 'department', 'job_id']
drop_cols = [c for c in drop_cols if c in df_clean.columns]
df_clean.drop(columns=drop_cols, inplace=True)
print(f"   🗑️  Drop kiye: {drop_cols}")

# Text columns — empty string se fill karo
text_fill = ['title', 'location', 'company_profile', 'description',
             'requirements', 'benefits', 'industry', 'function',
             'employment_type', 'required_experience', 'required_education']
for col in text_fill:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna('')

# Binary columns — 0 se fill karo
for col in ['telecommuting', 'has_company_logo', 'has_questions']:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna(0).astype(int)

# New feature : combined_text  (Phase 2 mein TF-IDF isme chalega)
df_clean['combined_text'] = (
    df_clean.get('title',           '') + ' ' +
    df_clean.get('company_profile', '') + ' ' +
    df_clean.get('description',     '') + ' ' +
    df_clean.get('requirements',    '') + ' ' +
    df_clean.get('benefits',        '')
).str.strip()

# New feature : text length (numeric feature for model)
for col in ['description', 'company_profile', 'requirements']:
    if col in df_clean.columns:
        df_clean[col + '_len'] = df_clean[col].apply(len)

df_clean.to_csv('fake_jobs_cleaned.csv', index=False)

print(f"\n   ✅ fake_jobs_cleaned.csv save ho gaya!")
print(f"   📊 Shape : {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
print(f"   🆕 New columns added:")
print(f"      - combined_text    (TF-IDF ke liye)")
print(f"      - description_len, company_profile_len, requirements_len")


# ──────────────────────────────────────────────
#  FINAL SUMMARY : Phase 1 Complete!
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("🎉 PHASE 1 COMPLETE — EDA Summary")
print("="*60)

print("""
╔══════════════════════════════════════════════════════════╗
║         TOP RED FLAGS JO EDA SE MILI                     ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  🚨 HIGH RISK signals:                                   ║
║     1. has_company_logo = 0  → Company logo nahi         ║
║     2. has_questions = 0     → Koi screening nahi        ║
║     3. description bahut     → Chhoti / vague details   ║
║        chhoti hai                                        ║
║                                                          ║
║  ⚠️  MEDIUM RISK signals:                                ║
║     4. telecommuting = 1     → WFH / remote job         ║
║     5. Part-time /           → Higher fraud rate        ║
║        Other employment type                             ║
║     6. Entry Level /         → Most targeted group      ║
║        Internship experience                             ║
║                                                          ║
║  🔤 TEXT RED FLAGS in title:                             ║
║     7. "data entry", "home   → Common scam titles       ║
║         based", "earn daily"                             ║
║     8. "no experience",      → Too good to be true      ║
║         "part time online"                               ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║  📁 Plots saved (7 total):                               ║
║     plot1_fake_vs_real.png                               ║
║     plot2_missing_values.png                             ║
║     plot3_binary_features.png                            ║
║     plot4_employment_type.png                            ║
║     plot5_text_lengths.png                               ║
║     plot6_scam_keywords.png                              ║
║     plot7_correlation.png                                ║
╠══════════════════════════════════════════════════════════╣
║  💾 fake_jobs_cleaned.csv → Phase 2 mein yahi use hoga   ║
║  ➡️  NEXT : Phase 2 — ML Model Training (Random Forest)  ║
╚══════════════════════════════════════════════════════════╝
""")
