# statistical_tests/parametric_tests/parametric.py
import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings("ignore")


def ttest_independent(
    group1: pd.Series,
    group2: pd.Series,
    equal_var: bool = True,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Two-sample independent t-test.

    Parameters
    ----------
    group1, group2 : pd.Series
        Two independent samples.
    equal_var : bool
        Assume equal variances (Student's t-test) or not (Welch's).
    alpha : float
        Significance level.

    Returns
    -------
    result : dict
        Contains t-statistic, p-value, degrees of freedom, Cohen's d, and decision.

    References
    ----------
    .. [1] Student (1908). The probable error of a mean. Biometrika.
    .. [2] Welch, B. L. (1947). The generalization of Student's problem when
           several different population variances are involved. Biometrika.
    """
    g1 = group1.dropna()
    g2 = group2.dropna()

    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=equal_var)
    n1, n2 = len(g1), len(g2)
    pooled_sd = np.sqrt(((n1 - 1) * g1.var() + (n2 - 1) * g2.var()) / (n1 + n2 - 2))
    cohens_d = (g1.mean() - g2.mean()) / pooled_sd if pooled_sd > 0 else np.nan

    return {
        "test": "Independent t-test",
        "t_statistic": t_stat,
        "p_value": p_val,
        "df": n1 + n2 - 2 if equal_var else np.nan,
        "cohens_d": cohens_d,
        "mean_group1": g1.mean(),
        "mean_group2": g2.mean(),
        "n1": n1,
        "n2": n2,
        "significant": p_val < alpha,
        "alpha": alpha
    }


def ttest_paired(
    before: pd.Series,
    after: pd.Series,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Paired t-test (dependent samples).

    Parameters
    ----------
    before, after : pd.Series
        Measurements before and after intervention.
    alpha : float

    Returns
    -------
    result : dict

    References
    ----------
    .. [1] Student (1908).
    """
    diff = (before - after).dropna()
    t_stat, p_val = stats.ttest_rel(before, after)

    return {
        "test": "Paired t-test",
        "t_statistic": t_stat,
        "p_value": p_val,
        "df": len(diff) - 1,
        "mean_difference": diff.mean(),
        "n": len(diff),
        "significant": p_val < alpha,
        "alpha": alpha
    }


def anova_oneway(
    groups: list,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    One-way ANOVA for k >= 2 independent groups.

    Parameters
    ----------
    groups : list of pd.Series or np.ndarray
    alpha : float

    Returns
    -------
    result : dict

    References
    ----------
    .. [1] Fisher, R. A. (1925). Statistical Methods for Research Workers.
    """
    clean_groups = [pd.Series(g).dropna() for g in groups]
    f_stat, p_val = stats.f_oneway(*clean_groups)

    return {
        "test": "One-way ANOVA",
        "f_statistic": f_stat,
        "p_value": p_val,
        "df_between": len(clean_groups) - 1,
        "df_within": sum(len(g) for g in clean_groups) - len(clean_groups),
        "n_groups": len(clean_groups),
        "significant": p_val < alpha,
        "alpha": alpha
    }