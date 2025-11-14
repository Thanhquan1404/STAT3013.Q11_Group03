import pandas as pd
from scipy import stats
from typing import Dict, Any

def run_kruskal(
    data: pd.DataFrame,
    value_col: str,
    label_col: str
) -> Dict[str, Any]:
    df = data[[value_col, label_col]].dropna()
    groups = [g[value_col].values for _, g in df.groupby(label_col)]

    H, p_value = stats.kruskal(*groups)

    n = sum(len(g) for g in groups)
    eps_sq = (H - len(groups) + 1) / (n - len(groups)) if n > len(groups) else 0

    return {
        "test": "Kruskal-Wallis",
        "p_value": p_value,
        "effect_size": eps_sq,
        "effect_size_type": "epsilon_squared"
    }
