# statistical_tests/correlation_tests/correlation.py
import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any


def pearson_correlation(
    x: pd.Series,
    y: pd.Series,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Pearson's product-moment correlation.

    References
    ----------
    .. [1] Pearson, K. (1895). Contributions to the mathematical theory of evolution.
    """
    x_clean = x.dropna()
    y_clean = y.dropna()
    common_idx = x_clean.index.intersection(y_clean.index)
    x_c = x_clean.loc[common_idx]
    y_c = y_clean.loc[common_idx]

    r, p_val = stats.pearsonr(x_c, y_c)

    return {
        "test": "Pearson r",
        "correlation": r,
        "p_value": p_val,
        "n": len(x_c),
        "r_squared": r**2,
        "significant": p_val < alpha,
        "alpha": alpha
    }


def spearman_correlation(
    x: pd.Series,
    y: pd.Series,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Spearman's rank-order correlation.

    References
    ----------
    .. [1] Spearman, C. (1904). The proof and measurement of association
           between two things. American Journal of Psychology.
    """
    x_clean = x.dropna()
    y_clean = y.dropna()
    common_idx = x_clean.index.intersection(y_clean.index)
    x_c = x_clean.loc[common_idx]
    y_c = y_clean.loc[common_idx]

    rho, p_val = stats.spearmanr(x_c, y_c)

    return {
        "test": "Spearman ρ",
        "correlation": rho,
        "p_value": p_val,
        "n": len(x_c),
        "significant": p_val < alpha,
        "alpha": alpha
    }


def kendall_tau(
    x: pd.Series,
    y: pd.Series,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Kendall's τ correlation.

    References
    ----------
    .. [1] Kendall, M. G. (1938). A new measure of rank correlation.
    """
    x_clean = x.dropna()
    y_clean = y.dropna()
    common_idx = x_clean.index.intersection(y_clean.index)
    x_c = x_clean.loc[common_idx]
    y_c = y_clean.loc[common_idx]

    tau, p_val = stats.kendalltau(x_c, y_c)

    return {
        "test": "Kendall τ",
        "correlation": tau,
        "p_value": p_val,
        "n": len(x_c),
        "significant": p_val < alpha,
        "alpha": alpha
    }