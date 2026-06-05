"""
supply_chain_risk_analysis.py
Supply Chain Risk Analysis Tool — Global/Cross-Industry
For use in data analysis internship projects.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. SAMPLE DATA
# ─────────────────────────────────────────────

SUPPLIERS = [
    {"supplier": "SinoChip Ltd",       "region": "China",       "category": "Semiconductors", "spend_pct": 18, "tier": 1},
    {"supplier": "VietFab Co.",         "region": "Vietnam",     "category": "Assembly",        "spend_pct": 14, "tier": 1},
    {"supplier": "SpeedFreight EU",     "region": "Netherlands", "category": "Logistics",       "spend_pct": 12, "tier": 1},
    {"supplier": "TechParts Mexico",    "region": "Mexico",      "category": "Components",      "spend_pct": 11, "tier": 1},
    {"supplier": "NordicRaw AB",        "region": "Sweden",      "category": "Raw Materials",   "spend_pct":  9, "tier": 2},
    {"supplier": "IndiaIT Services",    "region": "India",       "category": "IT/Software",     "spend_pct":  7, "tier": 2},
    {"supplier": "BrazilPack SA",       "region": "Brazil",      "category": "Packaging",       "spend_pct":  6, "tier": 2},
    {"supplier": "KoreaSteel Corp",     "region": "South Korea", "category": "Raw Materials",   "spend_pct":  5, "tier": 2},
    {"supplier": "UKLogistics PLC",     "region": "UK",          "category": "Logistics",       "spend_pct":  5, "tier": 1},
    {"supplier": "TurkeyTextile Ltd",   "region": "Turkey",      "category": "Components",      "spend_pct":  4, "tier": 2},
    {"supplier": "CanadaChemicals Inc", "region": "Canada",      "category": "Raw Materials",   "spend_pct":  4, "tier": 3},
    {"supplier": "GermanAuto GmbH",     "region": "Germany",     "category": "Components",      "spend_pct":  5, "tier": 1},
]

RISKS = [
    {"risk_id": "R01", "name": "Geopolitical — APAC chip sourcing", "category": "Geopolitical",    "likelihood": 4, "impact": 5, "detection_difficulty": 3, "response_time": 4, "dependency": 5},
    {"risk_id": "R02", "name": "Single-source logistics partner",    "category": "Logistics",       "likelihood": 3, "impact": 5, "detection_difficulty": 2, "response_time": 3, "dependency": 5},
    {"risk_id": "R03", "name": "Tier-2 supplier financial distress", "category": "Supplier",        "likelihood": 4, "impact": 4, "detection_difficulty": 4, "response_time": 3, "dependency": 4},
    {"risk_id": "R04", "name": "Climate — flood risk, Vietnam",      "category": "Climate",         "likelihood": 3, "impact": 4, "detection_difficulty": 2, "response_time": 2, "dependency": 4},
    {"risk_id": "R05", "name": "Ransomware on ERP systems",          "category": "Cyber",           "likelihood": 3, "impact": 4, "detection_difficulty": 4, "response_time": 3, "dependency": 3},
    {"risk_id": "R06", "name": "Demand spike — holiday season",      "category": "Demand",          "likelihood": 4, "impact": 3, "detection_difficulty": 2, "response_time": 2, "dependency": 3},
    {"risk_id": "R07", "name": "Port congestion — Rotterdam",        "category": "Logistics",       "likelihood": 4, "impact": 3, "detection_difficulty": 3, "response_time": 2, "dependency": 3},
    {"risk_id": "R08", "name": "Tariff increase — US/EU trade",      "category": "Geopolitical",    "likelihood": 3, "impact": 3, "detection_difficulty": 2, "response_time": 3, "dependency": 3},
    {"risk_id": "R09", "name": "Supplier quality failure",           "category": "Supplier",        "likelihood": 2, "impact": 4, "detection_difficulty": 3, "response_time": 2, "dependency": 4},
    {"risk_id": "R10", "name": "FX currency volatility",             "category": "Financial",       "likelihood": 4, "impact": 2, "detection_difficulty": 1, "response_time": 1, "dependency": 2},
]


# ─────────────────────────────────────────────
# 2. RISK SCORING ENGINE
# ─────────────────────────────────────────────

def compute_risk_scores(risks: list[dict]) -> pd.DataFrame:
    """
    Compute composite risk scores for a list of risk records.

    Scoring formula:
        base_score      = likelihood × impact           (range: 1–25)
        modifier        = mean(detection_difficulty,
                               response_time,
                               dependency) / 5          (range: 0.2–1.0)
        composite_score = base_score × modifier × 4    (normalised to ~100)

    Risk level thresholds:
        Critical  ≥ 70
        High      45–69
        Medium    20–44
        Low       < 20

    Parameters
    ----------
    risks : list of dict
        Each dict must contain the keys defined in RISKS above.

    Returns
    -------
    pd.DataFrame sorted descending by composite_score.
    """
    df = pd.DataFrame(risks)
    df["base_score"] = df["likelihood"] * df["impact"]
    df["modifier"]   = df[["detection_difficulty", "response_time", "dependency"]].mean(axis=1) / 5
    df["composite"]  = (df["base_score"] * df["modifier"] * 4).round().astype(int).clip(upper=100)

    def level(score):
        if score >= 70: return "Critical"
        if score >= 45: return "High"
        if score >= 20: return "Medium"
        return "Low"

    df["risk_level"] = df["composite"].apply(level)
    return df.sort_values("composite", ascending=False).reset_index(drop=True)


def score_suppliers(suppliers: list[dict], risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a supplier-level risk score derived from category-level risk averages
    and the supplier's spend concentration and tier.

    Parameters
    ----------
    suppliers : list of dict
    risk_df   : scored risk DataFrame from compute_risk_scores()

    Returns
    -------
    pd.DataFrame of suppliers with an added 'supplier_risk_score' column.
    """
    category_avg = risk_df.groupby("category")["composite"].mean()
    tier_penalty  = {1: 1.0, 2: 0.85, 3: 0.70}   # Tier 1 = full exposure

    sup_df = pd.DataFrame(suppliers)
    # Map logistics suppliers to the logistics category; others to supplier/geopolitical average
    def derive_score(row):
        cat_map = {
            "Logistics":      "Logistics",
            "Semiconductors": "Geopolitical",
            "IT/Software":    "Cyber",
            "Raw Materials":  "Climate",
        }
        base_cat = cat_map.get(row["category"], "Supplier")
        base     = category_avg.get(base_cat, category_avg.mean())
        spend_w  = 1 + (row["spend_pct"] - sup_df["spend_pct"].mean()) / 100
        return round(base * spend_w * tier_penalty.get(row["tier"], 0.8))

    sup_df["supplier_risk_score"] = sup_df.apply(derive_score, axis=1).clip(upper=100)

    def level(score):
        if score >= 70: return "Critical"
        if score >= 45: return "High"
        if score >= 20: return "Medium"
        return "Low"

    sup_df["risk_level"] = sup_df["supplier_risk_score"].apply(level)
    return sup_df.sort_values("supplier_risk_score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# 3. CONCENTRATION ANALYSIS
# ─────────────────────────────────────────────

def concentration_analysis(sup_df: pd.DataFrame) -> dict:
    """
    Compute supplier concentration metrics.

    Returns a dict with:
        top1_pct    : spend % of the single largest supplier
        top3_pct    : combined spend % of the top 3 suppliers
        hhi         : Herfindahl-Hirschman Index (0–10,000; >2500 = highly concentrated)
        single_region_pct : spend % dominated by the largest single region
    """
    spends = sup_df["spend_pct"].values / 100
    hhi    = round(sum((s * 100) ** 2 for s in spends))

    top3   = sup_df.nlargest(3, "spend_pct")["spend_pct"].sum()
    top1   = sup_df.nlargest(1, "spend_pct")["spend_pct"].values[0]

    region_spend = sup_df.groupby("region")["spend_pct"].sum()
    max_region   = region_spend.max()

    return {
        "top1_pct":           top1,
        "top3_pct":           top3,
        "hhi":                hhi,
        "single_region_pct":  max_region,
        "region_breakdown":   region_spend.sort_values(ascending=False),
    }


# ─────────────────────────────────────────────
# 4. MITIGATION RECOMMENDATIONS
# ─────────────────────────────────────────────

MITIGATION_PLAYBOOK = {
    "Critical": {
        "strategy":    "Avoid / Reduce immediately",
        "actions": [
            "Assign a dedicated risk owner within 48 hours",
            "Initiate dual-sourcing or alternative supplier qualification",
            "Increase safety stock to 60+ days for affected SKUs",
            "Escalate to senior leadership and board risk committee",
        ],
    },
    "High": {
        "strategy":    "Transfer / Reduce within 30 days",
        "actions": [
            "Review and update supplier contracts with penalty clauses",
            "Purchase supply-chain disruption insurance",
            "Identify and qualify at least one backup supplier",
            "Add to monthly risk dashboard with KPI thresholds",
        ],
    },
    "Medium": {
        "strategy":    "Reduce / Monitor quarterly",
        "actions": [
            "Schedule supplier audit within next quarter",
            "Define escalation trigger (e.g. lead time > 2× baseline)",
            "Review safety stock levels annually",
        ],
    },
    "Low": {
        "strategy":    "Accept — monitor annually",
        "actions": [
            "Document and include in annual risk register review",
            "No immediate action required",
        ],
    },
}

def get_recommendations(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach mitigation strategy and action summary to each risk.

    Parameters
    ----------
    risk_df : scored risk DataFrame

    Returns
    -------
    DataFrame with added 'strategy' and 'top_action' columns.
    """
    df = risk_df.copy()
    df["strategy"]   = df["risk_level"].map(lambda l: MITIGATION_PLAYBOOK[l]["strategy"])
    df["top_action"] = df["risk_level"].map(lambda l: MITIGATION_PLAYBOOK[l]["actions"][0])
    return df


# ─────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────

LEVEL_COLORS = {
    "Critical": "#E24B4A",
    "High":     "#EF9F27",
    "Medium":   "#378ADD",
    "Low":      "#639922",
}

def plot_risk_heatmap(risk_df: pd.DataFrame, save_path: str = "risk_heatmap.png"):
    """Plot a likelihood × impact heatmap with risk points overlaid."""
    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = LinearSegmentedColormap.from_list(
        "risk", ["#EAF3DE", "#FAEEDA", "#FAECE7", "#FCEBEB"], N=256
    )
    data = np.array([[l * i for i in range(1, 6)] for l in range(1, 6)], dtype=float)
    ax.imshow(data, cmap=cmap, origin="lower", vmin=1, vmax=25,
              extent=[0.5, 5.5, 0.5, 5.5], aspect="auto", alpha=0.85)

    for _, row in risk_df.iterrows():
        ax.scatter(row["likelihood"], row["impact"],
                   color=LEVEL_COLORS[row["risk_level"]], s=120, zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(row["risk_id"], (row["likelihood"], row["impact"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Likelihood (1 = rare → 5 = almost certain)", fontsize=10)
    ax.set_ylabel("Impact (1 = negligible → 5 = catastrophic)", fontsize=10)
    ax.set_title("Supply Chain Risk Heatmap", fontsize=13, fontweight="bold")
    ax.set_xticks(range(1, 6))
    ax.set_yticks(range(1, 6))
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)

    patches = [mpatches.Patch(color=c, label=l) for l, c in LEVEL_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_risk_bar(risk_df: pd.DataFrame, save_path: str = "risk_bar.png"):
    """Horizontal bar chart of composite risk scores, colour-coded by level."""
    df = risk_df.sort_values("composite")
    colors = [LEVEL_COLORS[l] for l in df["risk_level"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df["risk_id"] + " — " + df["name"], df["composite"],
                   color=colors, height=0.6, edgecolor="white")

    for bar, score in zip(bars, df["composite"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(score), va="center", fontsize=9)

    ax.set_xlabel("Composite risk score (0–100)", fontsize=10)
    ax.set_title("Risks Ranked by Composite Score", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.axvline(70, color="#E24B4A", linewidth=0.8, linestyle="--", alpha=0.6, label="Critical threshold")
    ax.axvline(45, color="#EF9F27", linewidth=0.8, linestyle="--", alpha=0.6, label="High threshold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_concentration(sup_df: pd.DataFrame, save_path: str = "concentration.png"):
    """Pie chart of supplier spend concentration."""
    top5  = sup_df.nlargest(5, "spend_pct")
    other = pd.DataFrame([{"supplier": "Others", "spend_pct": 100 - top5["spend_pct"].sum()}])
    plot_df = pd.concat([top5[["supplier", "spend_pct"]], other], ignore_index=True)

    colors = ["#E24B4A", "#EF9F27", "#378ADD", "#639922", "#9E9E9E", "#BDBDBD"]
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        plot_df["spend_pct"], labels=plot_df["supplier"],
        autopct="%1.0f%%", colors=colors, startangle=140,
        pctdistance=0.78, wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("Supplier Spend Concentration", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# 6. REPORT EXPORT
# ─────────────────────────────────────────────

def export_report(risk_df: pd.DataFrame, sup_df: pd.DataFrame,
                  concentration: dict, out_path: str = "risk_report.csv"):
    """
    Export the full risk register and supplier risk profile to CSV.

    Two sheets are written side-by-side as separate CSV files:
        risk_report.csv         — risk register with scores & recommendations
        supplier_risk_report.csv — supplier risk profile
    """
    rec_df = get_recommendations(risk_df)
    cols_risk = ["risk_id", "name", "category", "likelihood", "impact",
                 "composite", "risk_level", "strategy", "top_action"]
    rec_df[cols_risk].to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    sup_path = out_path.replace("risk_report", "supplier_risk_report")
    sup_df[["supplier", "region", "category", "tier", "spend_pct",
            "supplier_risk_score", "risk_level"]].to_csv(sup_path, index=False)
    print(f"  Saved: {sup_path}")

    print(f"\n  Concentration metrics:")
    print(f"    Top-1 supplier spend : {concentration['top1_pct']}%")
    print(f"    Top-3 supplier spend : {concentration['top3_pct']}%")
    print(f"    HHI index            : {concentration['hhi']} (>2500 = concentrated)")
    print(f"    Largest single region: {concentration['single_region_pct']}%")


# ─────────────────────────────────────────────
# 7. MAIN RUNNER
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Supply Chain Risk Analysis — Global/Cross-Industry")
    print("=" * 55)

    print("\n[1] Scoring risks...")
    risk_df = compute_risk_scores(RISKS)
    print(risk_df[["risk_id", "name", "composite", "risk_level"]].to_string(index=False))

    print("\n[2] Scoring suppliers...")
    sup_df = score_suppliers(SUPPLIERS, risk_df)
    print(sup_df[["supplier", "region", "spend_pct", "supplier_risk_score", "risk_level"]].to_string(index=False))

    print("\n[3] Concentration analysis...")
    concentration = concentration_analysis(sup_df)

    print("\n[4] Generating plots...")
    plot_risk_heatmap(risk_df)
    plot_risk_bar(risk_df)
    plot_concentration(sup_df)

    print("\n[5] Exporting report...")
    export_report(risk_df, sup_df, concentration)

    print("\n✓ Analysis complete.\n")


if __name__ == "__main__":
    main()
