"""
Replication: Cascade-Aware Prediction and Collaboration Routing
for Multi-Agent Software Development (PRICAI 2026)

Script 01: Prediction Models and Coefficient Analysis
  - Table 4: Nested feature sets (5-fold CV)
  - Section 5.2: Coefficient reversal analysis
  - Table 5: Feature importances from full cascade GB model
  - Cascade vs. agent-stratified comparison
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "../data/"

# ──────────────────────────────────────────────────────────────────────
# DATA LOADING AND FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

pr = pd.read_parquet(f"{DATA_DIR}pull_request.parquet")
commits = pd.read_parquet(f"{DATA_DIR}pr_commits.parquet")
task_type = pd.read_parquet(f"{DATA_DIR}pr_task_type.parquet")
reviews = pd.read_parquet(f"{DATA_DIR}pr_reviews.parquet")
commit_details = pd.read_parquet(f"{DATA_DIR}pr_commit_details.parquet",
                                  columns=["pr_id", "commit_stats_additions",
                                           "commit_stats_deletions"])

pr["merged"] = pr["merged_at"].notna().astype(int)

commit_stats = commits.groupby("pr_id").agg(
    n_commits=("sha", "count"),
    n_unique_authors=("author", "nunique"),
    n_unique_committers=("committer", "nunique"),
).reset_index()

def has_coauthor_trailer(messages):
    return any("co-authored-by" in str(m).lower() for m in messages if pd.notna(m))

coauth = commits.groupby("pr_id")["message"].apply(has_coauthor_trailer).reset_index()
coauth.columns = ["pr_id", "has_coauth"]

def get_collab_mode(group):
    authors = set(group["author"].dropna())
    committers = set(group["committer"].dropna())
    bot_patterns = ["[bot]", "github-actions", "dependabot", "copilot", "devin-ai",
                    "codex", "cursor", "claude"]
    def is_bot(name):
        return any(p in str(name).lower() for p in bot_patterns)
    has_human_author = any(not is_bot(a) for a in authors)
    has_human_committer = any(not is_bot(c) for c in committers)
    if has_human_author and has_human_committer:
        return "human_both"
    elif not has_human_author and has_human_committer:
        return "agent_draft"
    else:
        return "fully_autonomous"

collab_mode = commits.groupby("pr_id").apply(get_collab_mode).reset_index()
collab_mode.columns = ["pr_id", "collab_mode"]

task_type_map = task_type[["id", "type"]].rename(columns={"id": "pr_id"})

review_count = reviews.groupby("pr_id").agg(
    n_reviews=("id", "count"),
    has_changes_requested=("state", lambda x: int((x == "CHANGES_REQUESTED").any())),
).reset_index()

# Use max aggregation per PR for commit size (not sum)
size_stats = commit_details.groupby("pr_id").agg(
    total_additions=("commit_stats_additions", "max"),
    total_deletions=("commit_stats_deletions", "max"),
).reset_index()
size_stats["change_size"] = size_stats["total_additions"] + size_stats["total_deletions"]

df = pr[["id", "agent", "repo_id", "merged", "created_at"]].rename(columns={"id": "pr_id"})
df = df.merge(commit_stats, on="pr_id", how="left")
df = df.merge(coauth, on="pr_id", how="left")
df = df.merge(collab_mode, on="pr_id", how="left")
df = df.merge(task_type_map, on="pr_id", how="left")
df = df.merge(review_count, on="pr_id", how="left")
df = df.merge(size_stats[["pr_id", "change_size"]], on="pr_id", how="left")

df["n_commits"] = df["n_commits"].fillna(1)
df["has_coauth"] = df["has_coauth"].fillna(False).astype(int)
df["n_reviews"] = df["n_reviews"].fillna(0)
df["has_changes_requested"] = df["has_changes_requested"].fillna(0)
df["type"] = df["type"].fillna("other")
df["collab_mode"] = df["collab_mode"].fillna("fully_autonomous")
df["change_size"] = df["change_size"].fillna(df["change_size"].median())

le_agent = LabelEncoder()
df["agent_enc"] = le_agent.fit_transform(df["agent"])
le_task = LabelEncoder()
df["task_enc"] = le_task.fit_transform(df["type"])
le_collab = LabelEncoder()
df["collab_mode_enc"] = le_collab.fit_transform(df["collab_mode"])
df["log_commits"] = np.log1p(df["n_commits"])
df["log_size"] = np.log1p(df["change_size"])
repo_pr_count = df.groupby("repo_id")["merged"].count().to_dict()
df["repo_pr_count"] = df["repo_id"].map(repo_pr_count)
df["log_repo_prs"] = np.log1p(df["repo_pr_count"])

print(f"Dataset: {len(df)} PRs, {df['agent'].nunique()} agents")
print(f"Co-authorship rate: {df['has_coauth'].mean():.3f}")
print(f"Overall merge rate: {df['merged'].mean():.3f}")

y = df["merged"].values
n = len(df)

# Fill any remaining NaN in feature columns
feature_cols = ["has_coauth", "log_size", "n_reviews", "log_commits",
                "agent_enc", "log_repo_prs", "collab_mode_enc", "task_enc",
                "has_changes_requested", "n_unique_authors", "n_unique_committers"]
for col in feature_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# ──────────────────────────────────────────────────────────────────────
# TABLE 4: NESTED FEATURE SETS — 5-FOLD CROSS-VALIDATION
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLE 4: NESTED FEATURE SETS (5-FOLD CV)")
print("  Naive → Layer1 (+ agent) → Layer2 (+ repo) → Full cascade-aware")
print("=" * 70)

# Feature set definitions
naive_features = ["has_coauth", "log_size", "n_reviews", "log_commits"]
layer1_features = naive_features + ["agent_enc"]
layer2_features = layer1_features + ["log_repo_prs"]
full_features = layer2_features + ["collab_mode_enc", "task_enc",
                                    "has_changes_requested", "n_unique_authors",
                                    "n_unique_committers"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n  {'Feature Set':<35} {'LR AUC':>8} {'GB AUC':>8} {'GB Acc':>8}")
print(f"  {'-'*63}")

auc_results = {}
for name, feats in [
    ("Naive (has_coauth, log_size, n_reviews, log_commits)", naive_features),
    ("Layer 1: + agent identity",                           layer1_features),
    ("Layer 2: + repo context (log_repo_prs)",              layer2_features),
    ("Full cascade-aware",                                   full_features),
]:
    valid_feats = [f for f in feats if f in df.columns]
    X = df[valid_feats].values

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_probs = cross_val_predict(lr, X, y, cv=cv, method="predict_proba")[:, 1]
    lr_auc = roc_auc_score(y, lr_probs)

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    gb_probs = cross_val_predict(gb, X, y, cv=cv, method="predict_proba")[:, 1]
    gb_auc = roc_auc_score(y, gb_probs)
    gb_acc = accuracy_score(y, (gb_probs > 0.5).astype(int))

    auc_results[name] = {"lr_auc": lr_auc, "gb_auc": gb_auc, "gb_acc": gb_acc,
                         "feats": valid_feats, "gb_probs": gb_probs}
    print(f"  {name:<35} {lr_auc:.4f}   {gb_auc:.4f}   {gb_acc:.4f}")

naive_gb_auc = auc_results["Naive (has_coauth, log_size, n_reviews, log_commits)"]["gb_auc"]
full_gb_auc = auc_results["Full cascade-aware"]["gb_auc"]
print(f"\n  AUC gain Naive → Full: +{(full_gb_auc - naive_gb_auc)*100:.1f} ppAUC")

# ──────────────────────────────────────────────────────────────────────
# SECTION 5.2: COEFFICIENT REVERSAL ANALYSIS
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 5.2: COEFFICIENT REVERSAL")
print("  Naive LR vs. Layer1 LR vs. Full LR — has_coauth coefficient")
print("=" * 70)

for label, feats in [
    ("Naive LR (no agent context)", naive_features),
    ("Layer 1 LR (+ agent identity)", layer1_features),
    ("Full LR (all features)", full_features),
]:
    valid_feats = [f for f in feats if f in df.columns]
    X = df[valid_feats].values
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X, y)
    coef_idx = valid_feats.index("has_coauth")
    coef = lr.coef_[0][coef_idx]
    odds_ratio = np.exp(coef)
    print(f"\n  {label}")
    print(f"    has_coauth coefficient: {coef:+.4f}")
    print(f"    Odds ratio: {odds_ratio:.4f}")

print(f"\n  Interpretation: Sign and magnitude of has_coauth coefficient")
print(f"  changes across feature sets, illustrating omitted-variable bias.")

# ──────────────────────────────────────────────────────────────────────
# TABLE 5: FEATURE IMPORTANCES FROM FULL CASCADE GB MODEL
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLE 5: FEATURE IMPORTANCES — FULL CASCADE-AWARE GB MODEL")
print("=" * 70)

valid_full = [f for f in full_features if f in df.columns]
X_full = df[valid_full].values
gb_full = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
gb_full.fit(X_full, y)

importances = gb_full.feature_importances_
feat_importance = sorted(zip(valid_full, importances), key=lambda x: x[1], reverse=True)

print(f"\n  {'Feature':<30} {'Importance':>12}")
print(f"  {'-'*44}")
for feat, imp in feat_importance:
    print(f"  {feat:<30} {imp:.4f}")

# ──────────────────────────────────────────────────────────────────────
# CASCADE VS. AGENT-STRATIFIED COMPARISON
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CASCADE VS. AGENT-STRATIFIED COMPARISON")
print("  Pooled cascade model vs. separate per-agent GB models")
print("=" * 70)

# Per-agent stratified model: train separate GB per agent on naive features
all_probs_stratified = np.zeros(n)
for agent in df["agent"].unique():
    agent_mask = (df["agent"] == agent).values
    X_agent = df.loc[agent_mask, naive_features].values
    y_agent = y[agent_mask]
    if len(y_agent) < 50 or y_agent.sum() < 10 or (len(y_agent) - y_agent.sum()) < 10:
        all_probs_stratified[agent_mask] = y_agent.mean()
        continue
    gb_s = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs_s = cross_val_predict(gb_s, X_agent, y_agent, cv=cv_s, method="predict_proba")[:, 1]
    all_probs_stratified[agent_mask] = probs_s

auc_stratified = roc_auc_score(y, all_probs_stratified)

print(f"\n  {'Model':<45} {'AUC':>8}")
print(f"  {'-'*55}")
print(f"  {'Naive pooled GB (4 features)':<45} {naive_gb_auc:.4f}")
print(f"  {'Per-agent stratified GB (naive features)':<45} {auc_stratified:.4f}")
print(f"  {'Full cascade-aware pooled GB':<45} {full_gb_auc:.4f}")
print(f"\n  Full cascade advantage over per-agent stratified: "
      f"+{(full_gb_auc - auc_stratified)*100:.1f} ppAUC")
print(f"  Cascade vs. naive (same features, different pooling): "
      f"+{(full_gb_auc - naive_gb_auc)*100:.1f} ppAUC")

print("\n" + "=" * 70)
print("SCRIPT 01 COMPLETE")
print("=" * 70)
