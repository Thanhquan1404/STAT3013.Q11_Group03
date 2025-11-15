# statistical_tests/proportion_tests/proportion.py
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from typing import Tuple, Dict, Any
from scipy.stats import chi2_contingency

def proportion_test_two_samples(
    count1: int,
    n1: int,
    count2: int,
    n2: int,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> Dict[str, Any]:
    """
    Z-test for two independent proportions.

    Parameters
    ----------
    count1, count2 : int
        Number of successes.
    n1, n2 : int
        Sample sizes.
    alpha : float
    alternative : {'two-sided', 'smaller', 'larger'}

    Returns
    -------
    result : dict

    References
    ----------
    .. [1] Fleiss, J. L., et al. (2003). Statistical Methods for Rates and Proportions.
    """
    count = np.array([count1, count2])
    nobs = np.array([n1, n2])

    stat, p_val = proportions_ztest(count, nobs, alternative=alternative)
    prop1 = count1 / n1
    prop2 = count2 / n2
    diff = prop1 - prop2

    ci1 = proportion_confint(count1, n1, alpha=alpha, method='wilson')
    ci2 = proportion_confint(count2, n2, alpha=alpha, method='wilson')

    return {
        "test": "Two-Proportion Z-test",
        "proportion1": prop1,
        "proportion2": prop2,
        "difference": diff,
        "z_statistic": stat,
        "p_value": p_val,
        "ci_95_prop1": ci1,
        "ci_95_prop2": ci2,
        "n1": n1,
        "n2": n2,
        "significant": p_val < alpha,
        "alpha": alpha
    }


def chi2_contingency_test(
    contingency_table: pd.DataFrame,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Chi-squared test of independence.

    Parameters
    ----------
    contingency_table : pd.DataFrame
        2x2 or larger table (rows = group, cols = outcome).
    alpha : float

    Returns
    -------
    result : dict

    References
    ----------
    .. [1] Pearson, K. (1900). On the criterion that a given system of deviations...
    """
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)

    return {
        "test": "Chi-squared Test of Independence",
        "chi2_statistic": chi2,
        "p_value": p_val,
        "degrees_of_freedom": dof,
        "expected_frequencies": expected.tolist(),
        "significant": p_val < alpha,
        "alpha": alpha
    }