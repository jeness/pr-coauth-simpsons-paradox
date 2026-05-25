"""
Replication: Cascade-Aware Prediction and Collaboration Routing
for Multi-Agent Software Development (PRICAI 2026)

Script 03: Cost-Sensitive Collaboration Routing (CSCR)
  - CSCR algorithm: train outcome model, compute uplift, route by budget
  - Table 8: Policy comparison at B=0.30
  - Table 9: Per-agent CSCR on non-Codex subpopulation (temporal test split)
  - Budget sweep: B from 0 to 1
  - Ecosystem evolution: reweighted compositions
  - Cost-benefit analysis
"""
import pandas as pd
import numpy as np
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
n = len(df)

# ──────────────────────────────────────────────────────────────────────
# CSCR ALGORITHM
#
# Formulation: Given a human review budget B (fraction of PRs), find
# routing policy π that maximises expected merge rate:
#   max_π  E[M | π(x)]
#   s.t.   P(π(x) = "human") ≤ B
#
# Solution:
#   1. Train outcome model GB(200 trees, depth 5) on
#      [routing_features + collab_mode_enc]
#   2. Predict counterfactuals: p_auto = P(merge | x, fully_autonomous)
#                               p_human = P(merge | x, human_both)
#   3. Compute uplift = p_human - p_auto
#   4. Route top-B fraction by uplift to human review
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CSCR ALGORITHM: TRAINING OUTCOME MODEL ON FULL DATASET")
print("=" * 70)

features_routing = ["agent_enc", "task_enc", "log_size", "log_repo_prs", "log_commits"]
X_routing = df[features_routing].values
X_with_collab = np.column_stack([X_routing, df["collab_mode_enc"].values])

auto_code = le_collab.transform(["fully_autonomous"])[0]
human_code = le_collab.transform(["human_both"])[0]

# Train outcome model on full dataset
outcome_model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
outcome_model.fit(X_with_collab, y)

# Predict counterfactuals for all PRs
X_auto = np.column_stack([X_routing, np.full(n, auto_code)])
X_human = np.column_stack([X_routing, np.full(n, human_code)])
p_merge_auto = outcome_model.predict_proba(X_auto)[:, 1]
p_merge_human = outcome_model.predict_proba(X_human)[:, 1]
human_uplift = p_merge_human - p_merge_auto

df["p_merge_auto"] = p_merge_auto
df["p_merge_human"] = p_merge_human
df["human_uplift"] = human_uplift

print(f"\n  Human uplift statistics (full dataset):")
print(f"    Mean:   {human_uplift.mean()*100:.2f}pp")
print(f"    Median: {np.median(human_uplift)*100:.2f}pp")
print(f"    Std:    {human_uplift.std()*100:.2f}pp")
print(f"    % where human helps (uplift > 0): {(human_uplift > 0).mean()*100:.1f}%")

# ──────────────────────────────────────────────────────────────────────
# TABLE 8: POLICY COMPARISON AT B=0.30 (FULL DATASET)
# Policies: always-auto, random, agent-threshold, CSCR, risk-based, always-human
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLE 8: POLICY COMPARISON AT B=0.30")
print("=" * 70)

B = 0.30
n_human_budget = int(n * B)
mr_always_auto = p_merge_auto.mean()
mr_always_human = p_merge_human.mean()

# Policy: CSCR (uplift-based routing)
sorted_by_uplift = np.argsort(human_uplift)
policy_cscr = np.zeros(n, dtype=int)
policy_cscr[sorted_by_uplift[-n_human_budget:]] = 1
mr_cscr = np.where(policy_cscr == 1, p_merge_human, p_merge_auto).mean()

# Policy: Random allocation
rng42 = np.random.default_rng(42)
policy_random = np.zeros(n, dtype=int)
random_idx = rng42.choice(n, n_human_budget, replace=False)
policy_random[random_idx] = 1
mr_random = np.where(policy_random == 1, p_merge_human, p_merge_auto).mean()

# Policy: Agent-threshold (review agents with merge rate < 65%)
agent_mr = df.groupby("agent_enc")["merged"].mean()
low_mr_agents = agent_mr[agent_mr < 0.65].index
policy_threshold = df["agent_enc"].isin(low_mr_agents).astype(int).values
if policy_threshold.sum() > n_human_budget:
    thresh_indices = np.where(policy_threshold == 1)[0]
    keep = rng42.choice(thresh_indices, n_human_budget, replace=False)
    policy_threshold = np.zeros(n, dtype=int)
    policy_threshold[keep] = 1
mr_threshold = np.where(policy_threshold == 1, p_merge_human, p_merge_auto).mean()

# Policy: Risk-based (route lowest p_auto to human review)
sorted_by_auto_prob = np.argsort(p_merge_auto)
policy_risk = np.zeros(n, dtype=int)
policy_risk[sorted_by_auto_prob[:n_human_budget]] = 1
mr_risk = np.where(policy_risk == 1, p_merge_human, p_merge_auto).mean()

print(f"\n  {'Policy':<35} {'E[MR]':>8} {'Gain vs auto':>14} {'Budget':>8}")
print(f"  {'-'*68}")
print(f"  {'Always autonomous (B=0)':<35} {mr_always_auto*100:>6.2f}%  {'---':>12}  {'0%':>6}")
print(f"  {'Random allocation':<35} {mr_random*100:>6.2f}%  {(mr_random-mr_always_auto)*100:>+10.2f}pp  {'30%':>6}")
print(f"  {'Agent-threshold (MR<65%)':<35} {mr_threshold*100:>6.2f}%  {(mr_threshold-mr_always_auto)*100:>+10.2f}pp  {'30%':>6}")
print(f"  {'CSCR (uplift-based, ours)':<35} {mr_cscr*100:>6.2f}%  {(mr_cscr-mr_always_auto)*100:>+10.2f}pp  {'30%':>6}")
print(f"  {'Risk-based (lowest p_auto)':<35} {mr_risk*100:>6.2f}%  {(mr_risk-mr_always_auto)*100:>+10.2f}pp  {'30%':>6}")
print(f"  {'Always human (B=1)':<35} {mr_always_human*100:>6.2f}%  {(mr_always_human-mr_always_auto)*100:>+10.2f}pp  {'100%':>6}")

efficiency = (mr_cscr - mr_always_auto) / max(mr_always_human - mr_always_auto, 1e-9) * 100
print(f"\n  CSCR efficiency: {efficiency:.1f}% of max gain with only 30% budget")

# ──────────────────────────────────────────────────────────────────────
# TABLE 9: PER-AGENT CSCR ON NON-CODEX SUBPOPULATION
# Temporal test split: train < June 15, 2025 / test >= June 15, 2025
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLE 9: PER-AGENT CSCR — NON-CODEX SUBPOPULATION")
print("  Temporal test split: train < 2025-06-15, test >= 2025-06-15")
print("=" * 70)

df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
split_date = pd.Timestamp("2025-06-15", tz="UTC")
train_mask = df["created_at"] < split_date
test_mask = df["created_at"] >= split_date

print(f"\n  Train: {train_mask.sum()} PRs  |  Test: {test_mask.sum()} PRs")

# Retrain outcome model on training data only
X_train_collab = np.column_stack([
    df.loc[train_mask, features_routing].values,
    df.loc[train_mask, "collab_mode_enc"].values
])
y_train = y[train_mask]

outcome_model_temp = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
outcome_model_temp.fit(X_train_collab, y_train)

# Predict counterfactuals on test set
X_test_routing = df.loc[test_mask, features_routing].values
n_test = test_mask.sum()

X_auto_test = np.column_stack([X_test_routing, np.full(n_test, auto_code)])
X_human_test = np.column_stack([X_test_routing, np.full(n_test, human_code)])

p_auto_test = outcome_model_temp.predict_proba(X_auto_test)[:, 1]
p_human_test = outcome_model_temp.predict_proba(X_human_test)[:, 1]
uplift_test = p_human_test - p_auto_test

df_test = df[test_mask].copy()
df_test = df_test.reset_index(drop=True)
df_test["p_auto"] = p_auto_test
df_test["p_human"] = p_human_test
df_test["uplift"] = uplift_test

# Global CSCR routing at B=0.30 on test set
B = 0.30
n_human_test = int(len(df_test) * B)
uplift_all = df_test["uplift"].values
thresh_all = np.sort(uplift_all)[::-1][min(n_human_test - 1, len(df_test) - 1)]
routed_all = df_test["uplift"] >= thresh_all

# Resolve ties
if routed_all.sum() > n_human_test:
    tie_mask = df_test["uplift"] == thresh_all
    excess = routed_all.sum() - n_human_test
    tie_idx = df_test[tie_mask & routed_all].index
    rng_tie = np.random.default_rng(42)
    drop_idx = rng_tie.choice(tie_idx, size=min(excess, len(tie_idx)), replace=False)
    routed_all.loc[drop_idx] = False

# Non-Codex subset
codex_mask_test = df_test["agent"] == "OpenAI_Codex"
df_test_nc = df_test[~codex_mask_test].copy()
print(f"\n  Non-Codex test set: {len(df_test_nc)} PRs")

if len(df_test_nc) > 0:
    # CSCR at B=0.30 within non-Codex subset
    n_human_nc = int(len(df_test_nc) * B)
    uplift_nc = df_test_nc["uplift"].values
    thresh_nc = np.sort(uplift_nc)[::-1][min(n_human_nc - 1, len(df_test_nc) - 1)]
    routed_nc = (df_test_nc["uplift"] >= thresh_nc).copy()

    # Resolve ties
    if routed_nc.sum() > n_human_nc:
        tie_mask_nc = df_test_nc["uplift"] == thresh_nc
        excess_nc = routed_nc.sum() - n_human_nc
        tie_idx_nc = df_test_nc[tie_mask_nc & routed_nc].index
        rng_nc = np.random.default_rng(42)
        drop_nc = rng_nc.choice(tie_idx_nc, size=min(excess_nc, len(tie_idx_nc)), replace=False)
        routed_nc.loc[drop_nc] = False

    mr_auto_nc = df_test_nc["p_auto"].mean()
    mr_human_nc = df_test_nc["p_human"].mean()
    mr_cscr_nc = (
        df_test_nc.loc[routed_nc, "p_human"].sum()
        + df_test_nc.loc[~routed_nc, "p_auto"].sum()
    ) / len(df_test_nc)
    full_gain_nc = mr_human_nc - mr_auto_nc
    cscr_gain_nc = mr_cscr_nc - mr_auto_nc
    efficiency_nc = (cscr_gain_nc / full_gain_nc * 100) if full_gain_nc > 0 else 0.0

    print(f"  Non-Codex overall: MR_auto={mr_auto_nc*100:.1f}%, "
          f"MR_CSCR={mr_cscr_nc*100:.1f}%, "
          f"gain={cscr_gain_nc*100:+.1f}pp, "
          f"efficiency={efficiency_nc:.1f}%")

    print(f"\n  Per-agent breakdown (non-Codex, B=0.30):")
    print(f"  {'Agent':<20} {'n':>6} {'MR_auto':>8} {'MR_CSCR':>8} {'Gain':>8} {'% routed':>9}")
    print(f"  {'-'*62}")

    for agent in sorted(df_test_nc["agent"].unique()):
        a_df = df_test_nc[df_test_nc["agent"] == agent]
        if len(a_df) < 10:
            continue
        mr_a = a_df["p_auto"].mean()
        routed_a = routed_nc[a_df.index]
        mr_c = (
            a_df.loc[routed_a, "p_human"].sum()
            + a_df.loc[~routed_a, "p_auto"].sum()
        ) / len(a_df)
        pct = routed_a.mean()
        print(f"  {agent:<20} {len(a_df):>6} {mr_a*100:>7.1f}% "
              f"{mr_c*100:>7.1f}% {(mr_c-mr_a)*100:>+7.1f}pp {pct*100:>8.1f}%")

# ──────────────────────────────────────────────────────────────────────
# BUDGET SWEEP: B from 0 to 1 (full dataset, in-sample)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BUDGET SWEEP: EXPECTED MERGE RATE vs. HUMAN REVIEW BUDGET")
print("  Full dataset, in-sample CSCR policy")
print("=" * 70)

print(f"\n  {'Budget B':>10} {'E[MR] CSCR':>12} {'E[MR] random':>14} {'Gain CSCR':>11}")
print(f"  {'-'*50}")

for B_val in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    n_h = int(n * B_val)
    # CSCR policy
    sorted_idx = np.argsort(human_uplift)
    policy_b = np.zeros(n, dtype=int)
    if n_h > 0:
        policy_b[sorted_idx[-n_h:]] = 1
    mr_b = np.where(policy_b == 1, p_merge_human, p_merge_auto).mean()
    # Random policy
    policy_rand_b = np.zeros(n, dtype=int)
    if n_h > 0:
        rand_idx_b = np.random.default_rng(42).choice(n, n_h, replace=False)
        policy_rand_b[rand_idx_b] = 1
    mr_rand_b = np.where(policy_rand_b == 1, p_merge_human, p_merge_auto).mean()
    gain_b = mr_b - mr_always_auto
    print(f"  {B_val:>10.1f} {mr_b*100:>10.2f}%  {mr_rand_b*100:>12.2f}%  {gain_b*100:>+9.2f}pp")

# ──────────────────────────────────────────────────────────────────────
# ECOSYSTEM EVOLUTION: REWEIGHTED COMPOSITIONS
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ECOSYSTEM EVOLUTION: CSCR UNDER REWEIGHTED AGENT COMPOSITIONS")
print("  Temporal test set, B=0.30")
print("=" * 70)

def cscr_reweighted(test_df, weights, B=0.30, seed=42):
    """Compute CSCR expected merge rate under agent composition reweighting."""
    rng = np.random.default_rng(seed)
    n_sample = len(test_df)
    probs = weights / weights.sum()
    idx = rng.choice(n_sample, size=n_sample, replace=True, p=probs)
    sample = test_df.iloc[idx].reset_index(drop=True)

    mr_auto = sample["p_auto"].mean()
    mr_human = sample["p_human"].mean()
    n_h = int(len(sample) * B)
    if n_h == 0:
        return mr_auto, mr_auto, 0.0
    uplift_s = sample["uplift"].values
    thresh = np.sort(uplift_s)[::-1][min(n_h - 1, len(sample) - 1)]
    routed = sample["uplift"] >= thresh
    mr_cscr_val = (
        sample.loc[routed, "p_human"].sum()
        + sample.loc[~routed, "p_auto"].sum()
    ) / len(sample)
    return mr_auto, mr_cscr_val, mr_cscr_val - mr_auto

print(f"\n  {'Scenario':<35} {'MR_auto':>8} {'MR_CSCR':>9} {'Gain':>8}")
print(f"  {'-'*63}")

# Current composition (uniform weights)
w_current = np.ones(len(df_test))
mr_a, mr_c, gain = cscr_reweighted(df_test, w_current)
print(f"  {'Current composition':<35} {mr_a*100:>7.1f}% {mr_c*100:>8.1f}% {gain*100:>+7.1f}pp")

# Equal agent share
agents_test = df_test["agent"].unique()
w_equal = np.zeros(len(df_test))
for agent in agents_test:
    mask = (df_test["agent"] == agent).values
    n_a = mask.sum()
    if n_a > 0:
        w_equal[mask] = 1.0 / (len(agents_test) * n_a)
mr_a, mr_c, gain = cscr_reweighted(df_test, w_equal)
print(f"  {'Equal agent share':<35} {mr_a*100:>7.1f}% {mr_c*100:>8.1f}% {gain*100:>+7.1f}pp")

# Codex at 30%
codex_mask_test_arr = (df_test["agent"] == "OpenAI_Codex").values
non_codex_mask_arr = ~codex_mask_test_arr
n_codex_test = codex_mask_test_arr.sum()
n_other_test = non_codex_mask_arr.sum()
w_30 = np.ones(len(df_test))
if n_codex_test > 0 and n_other_test > 0:
    w_30[codex_mask_test_arr] = 0.30 / n_codex_test
    w_30[non_codex_mask_arr] = 0.70 / n_other_test
mr_a, mr_c, gain = cscr_reweighted(df_test, w_30)
print(f"  {'Codex at 30% (declining share)':<35} {mr_a*100:>7.1f}% {mr_c*100:>8.1f}% {gain*100:>+7.1f}pp")

# No Codex
df_test_no_codex = df_test[non_codex_mask_arr].copy().reset_index(drop=True)
w_nc2 = np.ones(len(df_test_no_codex))
mr_a, mr_c, gain = cscr_reweighted(df_test_no_codex, w_nc2)
print(f"  {'No Codex':<35} {mr_a*100:>7.1f}% {mr_c*100:>8.1f}% {gain*100:>+7.1f}pp")

# ──────────────────────────────────────────────────────────────────────
# COST-BENEFIT ANALYSIS
# Total cost(B) = B * n * c_h + (1 - E[MR|policy(B)]) * n * c_f
# Optimal budget B* minimises total cost for given c_f/c_h ratio
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COST-BENEFIT ANALYSIS")
print("  Total cost(B) = B * c_h + (1 - E[MR|CSCR(B)]) * c_f")
print("  Optimal B* minimises cost for each c_f/c_h ratio")
print("=" * 70)

# Precompute CSCR expected MR at each budget level
budget_grid = np.arange(0.0, 1.05, 0.05)
cscr_mr_by_budget = []
for B_val in budget_grid:
    n_h = int(n * B_val)
    sorted_idx = np.argsort(human_uplift)
    policy_b = np.zeros(n, dtype=int)
    if n_h > 0:
        policy_b[sorted_idx[-n_h:]] = 1
    mr_b = np.where(policy_b == 1, p_merge_human, p_merge_auto).mean()
    cscr_mr_by_budget.append((B_val, mr_b))

print(f"\n  {'c_f/c_h ratio':>14} {'Optimal B*':>12} {'E[MR] at B*':>14} {'Total cost':>12}")
print(f"  {'-'*55}")

for cost_ratio in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    best_cost = float('inf')
    best_B_star = 0.0
    best_mr_star = 0.0
    for B_val, mr_val in cscr_mr_by_budget:
        total_cost = B_val + cost_ratio * (1 - mr_val)
        if total_cost < best_cost:
            best_cost = total_cost
            best_B_star = B_val
            best_mr_star = mr_val
    print(f"  {cost_ratio:>14.1f} {best_B_star:>12.2f} {best_mr_star*100:>12.2f}%  {best_cost:>10.4f}")

print("\n" + "=" * 70)
print("SCRIPT 03 COMPLETE")
print("=" * 70)
