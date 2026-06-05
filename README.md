# Supply Chain Risk Analysis

A Python tool for identifying, scoring, and visualising supply chain risks across global, cross-industry supplier networks. Built for data analysis internship projects.

---

## What this tool does

| Step | Module | Output |
|------|--------|--------|
| Score individual risks | `compute_risk_scores()` | Ranked risk register with composite scores |
| Score suppliers | `score_suppliers()` | Supplier risk profile by spend & tier |
| Measure concentration | `concentration_analysis()` | HHI index, top-N spend %, regional breakdown |
| Generate visuals | `plot_*()` functions | Heatmap, bar chart, concentration pie chart |
| Export results | `export_report()` | Two CSV files ready for stakeholder review |

---

## Getting started

### Prerequisites

Python 3.11+ is recommended. Install dependencies:

```bash
pip install pandas numpy matplotlib
```

### Run the analysis

```bash
python supply_chain_risk_analysis.py
```

This runs the full pipeline on the built-in sample dataset and writes four output files to the working directory:

```
risk_heatmap.png          # Likelihood × impact scatter heatmap
risk_bar.png              # Risks ranked by composite score
concentration.png         # Supplier spend concentration pie chart
risk_report.csv           # Full risk register with recommendations
supplier_risk_report.csv  # Supplier risk profile
```

---

## How the scoring works

### Risk composite score (0–100)

```
base_score      = likelihood × impact               # 1–25
modifier        = mean(detection_difficulty,
                       response_time,
                       dependency) / 5              # 0.2–1.0
composite_score = base_score × modifier × 4        # clipped to 100
```

All inputs are rated 1–5:

| Field | 1 | 5 |
|-------|---|---|
| `likelihood` | Rare | Almost certain |
| `impact` | Negligible | Catastrophic |
| `detection_difficulty` | Obvious/monitored | Invisible until too late |
| `response_time` | Hours | Months |
| `dependency` | Easily replaced | No alternative |

### Risk levels

| Score | Level | Recommended action |
|-------|-------|--------------------|
| ≥ 70 | 🔴 Critical | Immediate escalation, dual-source within 30 days |
| 45–69 | 🟠 High | Mitigation plan within 30 days |
| 20–44 | 🔵 Medium | Quarterly monitoring, define trigger thresholds |
| < 20 | 🟢 Low | Accept, document, review annually |

### Supplier risk score

Supplier scores are derived from:
- Category-level risk averages (e.g. logistics suppliers inherit the logistics risk score)
- Spend concentration weight (higher spend = higher exposure)
- Tier penalty (Tier 1 = full exposure; Tier 3 = 70% weight)

---

## Using your own data

Replace `SUPPLIERS` and `RISKS` at the top of the file, or load from CSV:

```python
import pandas as pd

risks     = pd.read_csv("my_risks.csv").to_dict("records")
suppliers = pd.read_csv("my_suppliers.csv").to_dict("records")

risk_df = compute_risk_scores(risks)
sup_df  = score_suppliers(suppliers, risk_df)
```

### Required columns for risks

| Column | Type | Description |
|--------|------|-------------|
| `risk_id` | str | Unique identifier (e.g. `R01`) |
| `name` | str | Short description |
| `category` | str | Risk category (Geopolitical, Logistics, Supplier, Climate, Cyber, Demand, Financial) |
| `likelihood` | int 1–5 | Probability of occurrence |
| `impact` | int 1–5 | Severity if it occurs |
| `detection_difficulty` | int 1–5 | How hard to detect early |
| `response_time` | int 1–5 | How long to recover (1=fast, 5=slow) |
| `dependency` | int 1–5 | How reliant the business is on this area |

### Required columns for suppliers

| Column | Type | Description |
|--------|------|-------------|
| `supplier` | str | Supplier name |
| `region` | str | Country or region |
| `category` | str | Commodity/service category |
| `spend_pct` | float | Share of total procurement spend (%) |
| `tier` | int 1–3 | Supply chain tier (1 = direct) |

---

## Project structure

```
supply_chain_risk_analysis.py   # Main analysis file
README.md                       # This file
risk_report.csv                 # Generated — risk register
supplier_risk_report.csv        # Generated — supplier profile
risk_heatmap.png                # Generated — heatmap
risk_bar.png                    # Generated — ranked bar chart
concentration.png               # Generated — spend concentration
```

---

## Key concepts

**Herfindahl-Hirschman Index (HHI):** Measures supplier concentration. Calculated as the sum of squared market share percentages. Above 2,500 indicates high concentration — a supply chain vulnerability signal.

**Bullwhip effect:** Small fluctuations in consumer demand amplify exponentially upstream. A demand risk score above 12 suggests you should review forecast accuracy and safety stock policies.

**Tier mapping:** Tier 1 suppliers are your direct partners. Tier 2 are their suppliers. Most risk analyses under-invest in Tier 2+ visibility — but failures there cause the same downstream impact.

**Mitigation strategies:**
- **Avoid** — exit the risk (stop single-sourcing from a high-risk region)
- **Transfer** — insurance, financial hedging, penalty contracts
- **Reduce** — dual-sourcing, safety stock, nearshoring, cyber controls
- **Accept** — document low-score risks, review annually

---

## Extending the analysis

Some directions to explore for a deeper project:

- **Time series analysis:** Track risk scores quarterly to identify trending risks
- **Monte Carlo simulation:** Model financial impact distributions under different disruption scenarios
- **Network graph:** Map Tier 1–3 supplier relationships using `networkx`
- **External data enrichment:** Pull World Bank political stability scores, GDACS disaster alerts, or Altman Z-scores for supplier financial health
- **Dashboard:** Wrap the outputs in a Streamlit app for interactive stakeholder review

---

## Notes for the intern

- Always document your scoring assumptions. Stakeholders will ask.
- The heatmap is your most persuasive slide for leadership — keep it simple.
- A risk register is only useful if it's updated. Build a process for quarterly reviews.
- When in doubt, a risk is more likely to be under-scored than over-scored. Bias toward caution.

---

*Built for the data analysis internship programme. Extend freely.*
