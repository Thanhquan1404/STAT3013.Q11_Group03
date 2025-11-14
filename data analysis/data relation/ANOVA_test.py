import pandas as pd
from scipy import stats
from typing import Dict, Any

def ANOVA_test(
    data: pd.DataFrame,
    value_col: str,
    label_col: str
) -> Dict[str, Any]:
    df = data[[value_col, label_col]].dropna()

    groups = [g[value_col].values for _, g in df.groupby(label_col)]

    F, p_value = stats.f_oneway(*groups)

    overall_mean = df[value_col].mean()
    ss_between = sum(len(g) * (g.mean() - overall_mean)**2 for _, g in df.groupby(label_col))
    ss_total = ((df[value_col] - overall_mean) ** 2).sum()
    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    return {
        "test": "ANOVA",
        "p_value": p_value,
        "effect_size": eta_squared,
        "effect_size_type": "eta_squared"
    }
