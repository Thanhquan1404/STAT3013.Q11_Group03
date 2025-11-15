# statistical_tests/nonparametric_tests/nonparametric.py
import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any


def mann_whitney_u(
    group1: pd.Series,
    group2: pd.Series,
    alternative: str = "two-sided",
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Mann-Whitney U test (nonparametric independent samples).

    Parameters
    ----------
    group1, group2 : pd.Series
    alternative : {'two-sided', 'less', 'greater'}
    alpha : float

    Returns
    -------
    result : dict

    References
    ----------
    .. [1] Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of
           two random variables is stochastically larger than the other.
           Annals of Mathematical Statistics.
    """
    g1 = group1.dropna()
    g2 = group2.dropna()

    u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative=alternative)
    n1, n2 = len(g1), len(g2)
    r = abs(u_stat - n1 * n2 / 2) / (n1 * n2)

    return {
        "test": "Mann-Whitney U",
        "u_statistic": u_stat,
        "p_value": p_val,
        "rank_biserial_r": r,
        "n1": n1,
        "n2": n2,
        "alternative": alternative,
        "significant": p_val < alpha,
        "alpha": alpha
    }


def wilcoxon_signed_rank(
    before: pd.Series,
    after: pd.Series,
    alternative: str = "two-sided",
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test (paired nonparametric).

    References
    ----------
    .. [1] Wilcoxon, F. (1945). Individual comparisons by ranking methods.
    """
    diff = (before - after).dropna()
    if len(diff) == 0:
        return {"test": "Wilcoxon", "error": "No valid pairs"}

    stat, p_val = stats.wilcoxon(before, after, alternative=alternative)

    return {
        "test": "Wilcoxon Signed-Rank",
        "statistic": stat,
        "p_value": p_val,
        "n": len(diff),
        "significant": p_val < alpha,
        "alpha": alpha
    }


def kruskal_wallis(
    groups: list,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Kruskal-Wallis H test (nonparametric ANOVA).

    References
    ----------
    .. [1] Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in
           one-criterion variance analysis. JASA.
    """
    clean_groups = [pd.Series(g).dropna() for g in groups]
    h_stat, p_val = stats.kruskal(*clean_groups)

    return {
        "test": "Kruskal-Wallis H",
        "h_statistic": h_stat,
        "p_value": p_val,
        "df": len(clean_groups) - 1,
        "n_groups": len(clean_groups),
        "significant": p_val < alpha,
        "alpha": alpha
    }