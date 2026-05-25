# Replication Package: Cascade-Aware Prediction and Collaboration Routing for Multi-Agent Software Development

## Overview

This repository contains the analysis scripts and data processing code for the paper "Cascade-Aware Prediction and Collaboration Routing for Multi-Agent Software Development" submitted to PRICAI 2026.

## Data

The paper uses the **AIDev dataset** (Li et al., MSR 2026), containing 33,596 pull requests from five AI coding agents on open-source GitHub repositories.

Required parquet files (place in `data/` directory):
- `pull_request.parquet` — PR metadata (agent, repo, timestamps, merge status)
- `pr_commits.parquet` — Per-commit details (author, committer, messages)
- `pr_commit_details.parquet` — Commit statistics (additions, deletions)
- `pr_task_type.parquet` — Inferred task type per PR
- `pr_reviews.parquet` — Review events

## Scripts

### Core Analysis (produces all paper numbers)

| Script | Description | Paper Sections |
|--------|-------------|----------------|
| `01_prediction_and_coefficients.py` | Table 4 (cascade-aware prediction), coefficient reversal (Section 5.2), feature importance | Sections 5.1, 5.2, 5.3 |
| `02_causal_aipw.py` | AIPW estimation, trimmed sensitivity (Table 6), DAG sensitivity (3 specs), propensity diagnostics | Section 6 |
| `03_routing_cscr.py` | CSCR algorithm, policy comparison (Table 8), budget sweep (Figure 2), per-agent disaggregation (Table 9), ecosystem evolution | Section 7 |

### Supplementary

| Script | Description |
|--------|-------------|
| `04_supplementary.py` | Right-censoring sensitivity, cascade-vs-stratification comparison, review-count sensitivity |

## Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
scipy>=1.10
pyarrow>=12.0
```

Install: `pip install pandas numpy scikit-learn scipy pyarrow`

## Reproducing Results

```bash
# Place AIDev parquet files in data/ directory, then:
python 01_prediction_and_coefficients.py
python 02_causal_aipw.py
python 03_routing_cscr.py
python 04_supplementary.py
```

All scripts print results to stdout. Expected output matches paper tables and figures.

## License

Code is released under MIT License for review purposes.
Data is subject to the AIDev dataset license (see Li et al., MSR 2026).
