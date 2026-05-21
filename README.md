# Replication: The Human Factor: Simpson's Paradox in AI Agent Pull Request Co-Authorship

## Requirements

```bash
pip install -r requirements.txt
```

## Running

```bash
python 01_download_data.py       # Download AIDev data (~300MB)
python 02_simpsons_paradox.py    # RQ1 + RQ2: Simpson's Paradox + collaboration modes
python 03_rq3_did_regression.py  # RQ3: Difference-in-differences (multi-agent adoption)
```

## Expected Output

- Table 1: Pooled merge rates (coauth vs pure)
- Table 2: Per-agent Simpson's Paradox reversal
- Table 3: Author/Committer collaboration modes
- RQ3: DiD treatment effect = −12.1 pp (SE=3.6, p<0.001)
