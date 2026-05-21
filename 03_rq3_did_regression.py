"""RQ3: Difference-in-Differences analysis of multi-agent adoption.

Tests whether adopting a second agent changes weekly merge rates
within multi-agent repositories, using repo fixed effects and a
linear time trend.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

pr = pd.read_parquet("data/pull_request.parquet")
pr['merged'] = pr['merged_at'].notna().astype(int)
pr['created_at'] = pd.to_datetime(pr['created_at'])

repo_agents = pr.groupby('repo_url')['agent'].nunique()
multi_repos = set(repo_agents[repo_agents >= 2].index)
print(f"Multi-agent repos: {len(multi_repos)}")

multi_pr = pr[pr['repo_url'].isin(multi_repos)].copy()
print(f"PRs in multi-agent repos: {len(multi_pr)}")


def get_second_agent_date(group):
    agent_first = group.groupby('agent')['created_at'].min().sort_values()
    if len(agent_first) >= 2:
        return agent_first.iloc[1]
    return pd.NaT


repo_treatment_dates = multi_pr.groupby('repo_url').apply(get_second_agent_date).dropna()
multi_pr = multi_pr.merge(
    repo_treatment_dates.rename('treatment_date').reset_index(),
    on='repo_url', how='left'
)
multi_pr['post_treatment'] = (multi_pr['created_at'] >= multi_pr['treatment_date']).astype(int)
multi_pr['year_week'] = multi_pr['created_at'].dt.strftime('%Y-W%V')

# Aggregate to repo-week level
weekly = multi_pr.groupby(['repo_url', 'year_week']).agg(
    merge_rate=('merged', 'mean'),
    n_prs=('merged', 'count'),
    post_treatment=('post_treatment', 'max')
).reset_index()

all_weeks = sorted(weekly['year_week'].unique())
week_map = {w: i for i, w in enumerate(all_weeks)}
weekly['time_idx'] = weekly['year_week'].map(week_map)

print(f"Weekly panel: {len(weekly)} observations, {weekly['repo_url'].nunique()} repos")
print(f"Pre-treatment: {(weekly['post_treatment']==0).sum()}, Post-treatment: {(weekly['post_treatment']==1).sum()}")

# Within-transformation (equivalent to repo fixed effects)
weekly['mr_dm'] = weekly.groupby('repo_url')['merge_rate'].transform(lambda x: x - x.mean())
weekly['post_dm'] = weekly.groupby('repo_url')['post_treatment'].transform(lambda x: x - x.mean())
weekly['time_dm'] = weekly.groupby('repo_url')['time_idx'].transform(lambda x: x - x.mean())
weekly['repo_cat'] = pd.Categorical(weekly['repo_url']).codes

X = weekly[['post_dm', 'time_dm']]
y = weekly['mr_dm']
model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': weekly['repo_cat']})

print("\n" + "=" * 60)
print("DiD REGRESSION: weekly merge rates ~ post_treatment + time_trend")
print("  (repo fixed effects via within-transformation, clustered SE)")
print("=" * 60)
print(model.summary2().tables[1].to_string())

coef = model.params['post_dm']
se = model.bse['post_dm']
pval = model.pvalues['post_dm']
ci = model.conf_int().loc['post_dm']

print(f"\n{'='*60}")
print(f"PAPER RESULT (RQ3):")
print(f"  Treatment effect: {coef*100:.1f} pp")
print(f"  Standard error:   {se*100:.1f} pp")
print(f"  p-value:          {pval:.4f}")
print(f"  95% CI:           [{ci[0]*100:.1f}, {ci[1]*100:.1f}] pp")
print(f"{'='*60}")
