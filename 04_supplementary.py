"""
Replication: Cascade-Aware Prediction and Collaboration Routing
for Multi-Agent Software Development (PRICAI 2026)

Script 04: Supplementary Robustness Checks
  - Right-censoring sensitivity (exclude open PRs, recompute AUC)
  - Feature ablation: full cascade AUC excluding n_reviews
  - Temporal generalization (train pre-June 15, test post)
"""
import pandas as pd
import numpy as np
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

# Fill NaN in feature columns
full_features = [
    "has_coauth", "log_size", "n_reviews", "log_commits", "agent_enc",
    "log_repo_prs", "collab_mode_enc", "task_enc",
    "has_changes_requested", "n_unique_authors", "n_unique_committers"
]
for col in full_features:
    if col in df.columns:
        df[col] = df[col].fillna(0)

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Baseline: full cascade AUC on complete dataset
valid_full = [f for f in full_features if f in df.columns]
X_full = df[valid_full].values
gb_full = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
probs_full = cross_val_predict(gb_full, X_full, y, cv=cv5, method="predict_proba")[:, 1]
auc_full = roc_auc_score(y, probs_full)
print(f"\nBaseline full cascade AUC (all PRs): {auc_full:.4f}")

# ──────────────────────────────────────────────────────────────────────
# RIGHT-CENSORING SENSITIVITY
# Open PRs are coded as merged=0 (not merged yet). This checks whether
# excluding them changes the prediction AUC.
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RIGHT-CENSORING SENSITIVITY")
print("  Exclude open (potentially still pending) PRs, recompute AUC")
print("=" * 70)

n_open = ((pr["merged_at"].isna()) & (pr["state"] == "open")).sum()
n_total = len(pr)
print(f"\n  Open PRs (coded as merged=0): {n_open} ({100*n_open/n_total:.1f}% of all PRs)")

closed_ids = set(pr[pr["state"] != "open"]["id"])
df_closed = df[df["pr_id"].isin(closed_ids)].copy()
y_closed = df_closed["merged"].values
print(f"  Closed/merged PR subset: {len(df_closed)} PRs")

valid_feats_closed = [f for f in full_features if f in df_closed.columns]
X_closed = df_closed[valid_feats_closed].values
gb_closed = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
probs_closed = cross_val_predict(
    gb_closed, X_closed, y_closed,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    method="predict_proba"
)[:, 1]
auc_closed = roc_auc_score(y_closed, probs_closed)

print(f"\n  {'Subset':<40} {'n PRs':>8} {'AUC':>8}")
print(f"  {'-'*58}")
print(f"  {'All PRs (open coded as not merged)':<40} {len(df):>8} {auc_full:.4f}")
print(f"  {'Closed PRs only (excl. open)':<40} {len(df_closed):>8} {auc_closed:.4f}")
print(f"\n  AUC difference: {(auc_closed - auc_full)*100:+.1f} ppAUC")

if abs(auc_closed - auc_full) < 0.01:
    print("  Conclusion: Right-censoring has negligible impact (< 1 ppAUC)")
else:
    print(f"  Conclusion: Right-censoring shifts AUC by {(auc_closed - auc_full)*100:+.1f} ppAUC")

# ──────────────────────────────────────────────────────────────────────
# FEATURE ABLATION: EXCLUDING n_reviews
# n_reviews is a post-treatment variable (reviews happen after PR creation)
# and may introduce leakage. This checks AUC without it.
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FEATURE ABLATION: FULL CASCADE AUC EXCLUDING n_reviews")
print("  n_reviews is a post-treatment variable; this checks for leakage")
print("=" * 70)

full_no_reviews = [f for f in valid_full if f != "n_reviews"]
X_no_reviews = df[full_no_reviews].values
gb_no_reviews = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
probs_no_reviews = cross_val_predict(
    gb_no_reviews, X_no_reviews, y,
    cv=cv5, method="predict_proba"
)[:, 1]
auc_no_reviews = roc_auc_score(y, probs_no_reviews)

print(f"\n  {'Configuration':<45} {'AUC':>8}")
print(f"  {'-'*55}")
print(f"  {'Full cascade (with n_reviews)':<45} {auc_full:.4f}")
print(f"  {'Full cascade (without n_reviews)':<45} {auc_no_reviews:.4f}")
print(f"\n  AUC change from removing n_reviews: {(auc_no_reviews - auc_full)*100:+.1f} ppAUC")

# ──────────────────────────────────────────────────────────────────────
# TEMPORAL GENERALIZATION
# Train on PRs before June 15, 2025; test on PRs after.
# Evaluates out-of-time predictive performance.
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TEMPORAL GENERALIZATION")
print("  Train: PRs before 2025-06-15  |  Test: PRs after 2025-06-15")
print("=" * 70)

df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
split_date = pd.Timestamp("2025-06-15", tz="UTC")
train_mask = df["created_at"] < split_date
test_mask = df["created_at"] >= split_date

print(f"\n  Train: {train_mask.sum()} PRs  |  Test: {test_mask.sum()} PRs")
print(f"  Train merge rate: {y[train_mask].mean():.3f}  |  "
      f"Test merge rate: {y[test_mask].mean():.3f}")

# Naive features (no agent context)
naive_features = ["has_coauth", "log_size", "n_reviews", "log_commits"]
X_train_naive = df.loc[train_mask, naive_features].values
X_test_naive = df.loc[test_mask, naive_features].values
y_train = y[train_mask]
y_test = y[test_mask]

gb_naive_temp = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
gb_naive_temp.fit(X_train_naive, y_train)
preds_naive_test = gb_naive_temp.predict_proba(X_test_naive)[:, 1]
auc_naive_temp = roc_auc_score(y_test, preds_naive_test)
acc_naive_temp = accuracy_score(y_test, (preds_naive_test > 0.5).astype(int))

# Full cascade-aware features
X_train_full = df.loc[train_mask, valid_full].values
X_test_full = df.loc[test_mask, valid_full].values

gb_full_temp = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
gb_full_temp.fit(X_train_full, y_train)
preds_full_test = gb_full_temp.predict_proba(X_test_full)[:, 1]
auc_full_temp = roc_auc_score(y_test, preds_full_test)
acc_full_temp = accuracy_score(y_test, (preds_full_test > 0.5).astype(int))

print(f"\n  {'Model':<40} {'AUC (temporal)':>16} {'Acc (temporal)':>16}")
print(f"  {'-'*74}")
print(f"  {'Naive GB (4 features)':<40} {auc_naive_temp:>16.4f} {acc_naive_temp:>16.4f}")
print(f"  {'Full cascade-aware GB':<40} {auc_full_temp:>16.4f} {acc_full_temp:>16.4f}")
print(f"\n  Cascade advantage (temporal): +{(auc_full_temp - auc_naive_temp)*100:.1f} ppAUC")

# Compare temporal vs. cross-validation AUC
print(f"\n  Comparison of evaluation protocols:")
print(f"  {'Metric':<45} {'Naive':>8} {'Full':>8}")
print(f"  {'-'*63}")
print(f"  {'5-fold CV AUC (script 01)':<45} {'N/A':>8} {auc_full:>8.4f}")
print(f"  {'Temporal out-of-time AUC':<45} {auc_naive_temp:>8.4f} {auc_full_temp:>8.4f}")

print("\n" + "=" * 70)
print("SCRIPT 04 COMPLETE")
print("=" * 70)
