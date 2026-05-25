"""
Replication: Cascade-Aware Prediction and Collaboration Routing
for Multi-Agent Software Development (PRICAI 2026)

Script 02: Causal Estimation — AIPW
  - AIPW estimation of co-authorship effect on merge rate
  - Table 6: Trimmed AIPW sensitivity analysis
  - DAG sensitivity: three propensity specifications
  - Propensity score overlap diagnostics
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
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
treatment = df["has_coauth"].values
n = len(df)

# ──────────────────────────────────────────────────────────────────────
# AIPW ESTIMATION
# Treatment: has_coauth (co-authorship trailer present)
# Outcome: merged (PR merged = 1)
#
# Propensity model: LogisticRegression on [A, log_commits, log_size, T, R]
# Outcome model: GradientBoostingClassifier (100 trees, depth 4) on
#                [propensity features + treatment]
# AIPW formula with propensity clipped at [0.01, 0.99]
# Bootstrap CI: 1000 samples, seed 42
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("AIPW ESTIMATION")
print("  Treatment: has_coauth  |  Outcome: merged")
print("=" * 70)

# Propensity model
X_psm = df[["agent_enc", "log_commits", "log_size", "task_enc", "log_repo_prs"]].values
ps_model = LogisticRegression(max_iter=1000, random_state=42)
ps_model.fit(X_psm, treatment)
propensity = ps_model.predict_proba(X_psm)[:, 1]

# Outcome model: single model with treatment as feature
X_out = np.column_stack([X_psm, treatment])
out_model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
out_model.fit(X_out, y)

X_treat = X_out.copy(); X_treat[:, -1] = 1
X_ctrl = X_out.copy(); X_ctrl[:, -1] = 0
mu1 = out_model.predict_proba(X_treat)[:, 1]
mu0 = out_model.predict_proba(X_ctrl)[:, 1]

def compute_aipw_trimmed(Y, T, mu1, mu0, propensity, trim_lo=0.01, trim_hi=0.99):
    """Compute AIPW-ATE with trimming (exclude observations outside [trim_lo, trim_hi])."""
    mask = (propensity >= trim_lo) & (propensity <= trim_hi)
    n_trimmed = (~mask).sum()

    Y_t = Y[mask]
    T_t = T[mask]
    mu1_t = mu1[mask]
    mu0_t = mu0[mask]
    ps_t = np.clip(propensity[mask], trim_lo, trim_hi)
    n_valid = mask.sum()

    ate = np.mean(
        mu1_t - mu0_t
        + T_t * (Y_t - mu1_t) / ps_t
        - (1 - T_t) * (Y_t - mu0_t) / (1 - ps_t)
    )

    # Bootstrap CI (1000 samples, seed 42)
    rng = np.random.default_rng(42)
    boot_ates = []
    for _ in range(1000):
        idx = rng.integers(0, n_valid, size=n_valid)
        ate_b = np.mean(
            mu1_t[idx] - mu0_t[idx]
            + T_t[idx] * (Y_t[idx] - mu1_t[idx]) / ps_t[idx]
            - (1 - T_t[idx]) * (Y_t[idx] - mu0_t[idx]) / (1 - ps_t[idx])
        )
        boot_ates.append(ate_b)
    ci_lo = np.percentile(boot_ates, 2.5)
    ci_hi = np.percentile(boot_ates, 97.5)

    return ate, ci_lo, ci_hi, n_trimmed, n_valid

# Main AIPW estimate
ate_main, ci_lo_main, ci_hi_main, _, _ = compute_aipw_trimmed(
    y, treatment, mu1, mu0, propensity, 0.01, 0.99
)
print(f"\nAIPW-ATE (clip [0.01, 0.99], n={n}):")
print(f"  ATE = {ate_main*100:+.2f}pp")
print(f"  95% Bootstrap CI: [{ci_lo_main*100:+.2f}, {ci_hi_main*100:+.2f}]pp")
print(f"  CI spans zero: {'YES' if ci_lo_main <= 0 <= ci_hi_main else 'NO'}")

# ──────────────────────────────────────────────────────────────────────
# TABLE 6: TRIMMED AIPW SENSITIVITY
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLE 6: TRIMMED AIPW — SENSITIVITY TO PROPENSITY TRIMMING")
print("  Observations with propensity outside bounds are excluded")
print("=" * 70)

print(f"\n  {'Trim bounds':<15} {'n retained':>10} {'n excluded':>10} {'ATE (pp)':>10} {'95% CI':>22}")
print(f"  {'-'*70}")

for lo, hi in [(0.01, 0.99), (0.02, 0.98), (0.05, 0.95), (0.10, 0.90)]:
    ate_t, ci_lo_t, ci_hi_t, n_trim, n_valid = compute_aipw_trimmed(
        y, treatment, mu1, mu0, propensity, lo, hi
    )
    print(f"  [{lo:.2f}, {hi:.2f}]      {n_valid:>10} {n_trim:>10}   {ate_t*100:>+8.2f}pp   "
          f"[{ci_lo_t*100:+.2f}, {ci_hi_t*100:+.2f}]")

# ──────────────────────────────────────────────────────────────────────
# DAG SENSITIVITY: THREE PROPENSITY SPECIFICATIONS
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DAG SENSITIVITY: THREE PROPENSITY SPECIFICATIONS")
print("  Spec 1: Minimal (A, S, R) — agent, size, repo")
print("  Spec 2: Full (A, T, S, R) — + task type [paper specification]")
print("  Spec 3: Extended (A, T, S, R + T×A interaction)")
print("=" * 70)

# Spec 1: Minimal (A, S, R)
X_ps1 = df[["agent_enc", "log_commits", "log_size", "log_repo_prs"]].values
ps1 = LogisticRegression(max_iter=1000, random_state=42)
ps1.fit(X_ps1, treatment)
prop1 = ps1.predict_proba(X_ps1)[:, 1]

# Spec 2: Full (A, T, S, R) — paper specification
prop2 = propensity

# Spec 3: Extended (A, T, S, R + T×A interaction)
X_ps3 = np.column_stack([X_psm, df["task_enc"].values * df["agent_enc"].values])
ps3 = LogisticRegression(max_iter=1000, random_state=42)
ps3.fit(X_ps3, treatment)
prop3 = ps3.predict_proba(X_ps3)[:, 1]

print(f"\n  {'Specification':<35} {'ATE (pp)':>10} {'95% CI':>22}")
print(f"  {'-'*70}")

for spec_name, prop in [
    ("Minimal (A, S, R)",           prop1),
    ("Full (A, T, S, R) [paper]",   prop2),
    ("Extended (A, T, S, R + T×A)", prop3),
]:
    ps_clipped = np.clip(prop, 0.01, 0.99)
    ate_s = np.mean(
        mu1 - mu0
        + treatment * (y - mu1) / ps_clipped
        - (1 - treatment) * (y - mu0) / (1 - ps_clipped)
    )
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        b = np.mean(
            mu1[idx] - mu0[idx]
            + treatment[idx] * (y[idx] - mu1[idx]) / ps_clipped[idx]
            - (1 - treatment[idx]) * (y[idx] - mu0[idx]) / (1 - ps_clipped[idx])
        )
        boot.append(b)
    ci_l, ci_h = np.percentile(boot, [2.5, 97.5])
    print(f"  {spec_name:<35} {ate_s*100:>+8.2f}pp   [{ci_l*100:+.2f}, {ci_h*100:+.2f}]")

# ──────────────────────────────────────────────────────────────────────
# PROPENSITY SCORE OVERLAP DIAGNOSTICS
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PROPENSITY SCORE OVERLAP DIAGNOSTICS")
print("=" * 70)

treated_ps = propensity[treatment == 1]
control_ps = propensity[treatment == 0]

print(f"\n  Treated (has_coauth=1, n={len(treated_ps)}):")
print(f"    Mean={treated_ps.mean():.4f}, Median={np.median(treated_ps):.4f}")
print(f"    [Q1={np.percentile(treated_ps, 25):.4f}, Q3={np.percentile(treated_ps, 75):.4f}]")
print(f"    Min={treated_ps.min():.4f}, Max={treated_ps.max():.4f}")

print(f"\n  Control (has_coauth=0, n={len(control_ps)}):")
print(f"    Mean={control_ps.mean():.4f}, Median={np.median(control_ps):.4f}")
print(f"    [Q1={np.percentile(control_ps, 25):.4f}, Q3={np.percentile(control_ps, 75):.4f}]")
print(f"    Min={control_ps.min():.4f}, Max={control_ps.max():.4f}")

extreme_lo = (propensity < 0.01).sum()
extreme_hi = (propensity > 0.99).sum()
print(f"\n  Extreme propensity scores:")
print(f"    < 0.01: {extreme_lo} ({100*extreme_lo/n:.1f}%)")
print(f"    > 0.99: {extreme_hi} ({100*extreme_hi/n:.1f}%)")
print(f"    Total extreme: {extreme_lo + extreme_hi} ({100*(extreme_lo+extreme_hi)/n:.1f}%)")

print("\n" + "=" * 70)
print("SCRIPT 02 COMPLETE")
print("=" * 70)
