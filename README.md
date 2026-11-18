# Sample analysis and dummy data for a research proposal

**All data in this repository are synthetic or use assumed parameters. Nothing here is a result of the proposed study.**

Supporting material for the proposal *Simulating AI Data Centers as Frequency Reserves in Qatar* (Appendix A.3).

## Run

```
pip install -r requirements.txt
python3 sample_analysis.py .
```

Python 3.10 or later. The random seed is fixed (2026), so every run returns the same numbers.

## Files

| File | Content | In the proposal |
|---|---|---|
| `sample_analysis.py` | Script: Parts 1 to 5, then CSV export | Appendix A.3 |
| `sample_analysis_output.txt` | Printed output of one run, including the run ledger | Appendix A.3, Table A9 |
| `fig3_sfr.png` | Illustrative frequency response | Figure 3 |
| `dummy_data/fig3_frequency_traces.csv` | Frequency every 10 ms for 30 s after a 400 MW trip, four fleet sizes | Figure 3 |
| `dummy_data/fig3_summary.csv` | Nadir, RoCoF and load-shedding stages per fleet | Appendix A.3 |
| `dummy_data/table10_paired_nadir_differences.csv` | 600 dummy paired differences (4 comparisons, 15 days, 10 seeds) | Table 10 |
| `dummy_data/table10_results.csv` | Estimates, 95% intervals, rank-biserial r, Holm p | Table 10 |
| `dummy_data/coding_reliability_segments.csv` | 60 double-coded interview segments | Cohen's kappa 0.895 |
| `dummy_data/variance_pilot_day_by_scenario.csv` | 15 days by 36 scenarios | Variance pilot |

The power simulation of Table A6 (4,000 replicates per row) runs inside the script and is summarised in the output.

The code was written with the help of an AI assistant (Claude, Anthropic), as declared in the proposal.

## Licence

Code: MIT. Data: CC BY 4.0.
