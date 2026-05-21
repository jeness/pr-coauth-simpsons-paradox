"""Simpson's Paradox in AI Agent PR Co-Authorship.

Reproduces all results:
- Co-Authored-By prevalence
- Pooled vs per-agent merge rate (Simpson's Paradox)
- Author/Committer collaboration modes
- Multi-agent repo adoption (null result)
"""
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load data
pr_commits = pd.read_parquet("data/pr_commits.parquet")
pr = pd.read_parquet("data/pull_request.parquet")

# === Per-PR Co-Authored-By flag ===
coauth_flag = pr_commits.groupby("pr_id")["message"].apply(
    lambda msgs: any("Co-Authored-By:" in str(m) or "Co-authored-by:" in str(m) for m in msgs)
).reset_index()
coauth_flag.columns = ["pr_id", "has_coauth"]

combined = pr[["id", "agent", "merged_at"]].rename(columns={"id": "pr_id"}).copy()
combined["merged"] = combined["merged_at"].notna()
combined = combined.merge(coauth_flag, on="pr_id", how="left")
combined["has_coauth"] = combined["has_coauth"].fillna(False)

print("=" * 60)
print("RESULT 1: Co-Authored-By Prevalence")
print("=" * 60)

coauth_n = int(combined["has_coauth"].sum())
total = len(combined)
print(f"Total PRs: {total}")
print(f"With Co-Authored-By: {coauth_n} ({coauth_n/total*100:.1f}%)")
print(f"Without: {total - coauth_n} ({(total-coauth_n)/total*100:.1f}%)")

# === RESULT 2: Pooled Simpson's Paradox ===
print("\n" + "=" * 60)
print("RESULT 2: Simpson's Paradox (Pooled vs Per-Agent)")
print("=" * 60)

coauth_group = combined[combined["has_coauth"] == True]
pure_group = combined[combined["has_coauth"] == False]

mr_coauth = coauth_group["merged"].mean()
mr_pure = pure_group["merged"].mean()
print(f"\nPOOLED:")
print(f"  With Co-Authored-By: {mr_coauth:.3f} (n={len(coauth_group)})")
print(f"  Without:             {mr_pure:.3f} (n={len(pure_group)})")
print(f"  Delta: {mr_coauth - mr_pure:+.3f}")

ct = pd.crosstab(combined["has_coauth"], combined["merged"])
chi2, p, _, _ = chi2_contingency(ct)
print(f"  Chi-square: {chi2:.1f}, p={p:.2e}")

print(f"\nPER-AGENT (reversal):")
print(f"{'Agent':<14} {'%coauth':<9} {'MR(coauth)':<12} {'MR(pure)':<10} {'Delta':<8} {'n_co'}")
print("-" * 61)
for agent in ["OpenAI_Codex", "Copilot", "Cursor", "Devin", "Claude_Code"]:
    ag = combined[combined["agent"] == agent]
    ag_co = ag[ag["has_coauth"] == True]
    ag_pu = ag[ag["has_coauth"] == False]
    pct = len(ag_co) / len(ag) * 100 if len(ag) > 0 else 0
    mr_c = ag_co["merged"].mean() if len(ag_co) > 0 else float("nan")
    mr_p = ag_pu["merged"].mean() if len(ag_pu) > 0 else float("nan")
    delta = mr_c - mr_p if not (np.isnan(mr_c) or np.isnan(mr_p)) else float("nan")
    print(f"{agent:<14} {pct:<9.1f} {mr_c:<12.3f} {mr_p:<10.3f} {delta:+.3f}   {len(ag_co)}")

# === RESULT 3: Author/Committer Modes ===
print("\n" + "=" * 60)
print("RESULT 3: Author/Committer Collaboration Modes")
print("=" * 60)

known_bots = ["copilot", "devin", "cursor", "claude", "codex", "bot", "cursoragent", "github-actions"]

def is_bot(name):
    return any(b in str(name).lower() for b in known_bots)

pr_commits["bot_author"] = pr_commits["author"].apply(is_bot)
pr_commits["bot_committer"] = pr_commits["committer"].apply(is_bot)

def mode_label(row):
    if row["bot_author"] and row["bot_committer"]:
        return "fully_auto"
    elif row["bot_author"] and not row["bot_committer"]:
        return "agent_draft"
    elif not row["bot_author"] and row["bot_committer"]:
        return "human_draft_agent_commit"
    else:
        return "human_both"

pr_commits["mode"] = pr_commits.apply(mode_label, axis=1)

pr_dom = pr_commits.groupby("pr_id")["mode"].agg(lambda x: x.mode().iloc[0]).reset_index()
pr_dom.columns = ["pr_id", "dom_mode"]
pr_dom = pr_dom.merge(combined[["pr_id", "merged"]], on="pr_id", how="left")

print(f"\n{'Mode':<28} {'Merge%':<10} {'n'}")
print("-" * 44)
for mode in ["human_both", "agent_draft", "fully_auto", "human_draft_agent_commit"]:
    sub = pr_dom[pr_dom["dom_mode"] == mode]
    if len(sub) >= 10:
        print(f"{mode:<28} {sub['merged'].mean():<10.3f} {len(sub)}")

# === RESULT 4: Multi-Agent Repos ===
print("\n" + "=" * 60)
print("RESULT 4: Multi-Agent Repo Adoption")
print("=" * 60)

repo_agents = pr.groupby("repo_url")["agent"].nunique()
multi_repos = repo_agents[repo_agents >= 2]
print(f"Multi-agent repos: {len(multi_repos)}")
print(f"See 03_rq3_did_regression.py for full DiD analysis")
