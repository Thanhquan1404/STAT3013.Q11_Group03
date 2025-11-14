import pandas as pd
from scipy import stats

def t_test(df: pd.DataFrame, column_name = "column name", column_label = "column label"):
    df_clean = df.dropna(subset=[column_name, column_label])

    normal_data = df_clean[df_clean[column_label] == 1][column_name]
    abnormal_data = df_clean[df_clean[column_label] == 2][column_name]

    n_normal = len(normal_data)
    n_abnormal = len(abnormal_data)

    if n_normal == 0 or n_abnormal == 0:
        print(f"⚠️ Not enough data to perform t-test for column '{column_name}'.")
        return

    t_stat, p_value = stats.ttest_ind(normal_data, abnormal_data, equal_var=False)

    print(f"="*30 + " T-Test Result for Clinical Variable: {column_name} " + "="*30 )
    print("-"*60)
    print(f"Normal group size: {n_normal}")
    print(f"Abnormal group size: {n_abnormal}")
    print(f"Mean (Normal):   {normal_data.mean():.4f}")
    print(f"Mean (Abnormal): {abnormal_data.mean():.4f}")
    print(f"T-statistic:     {t_stat:.4f}")
    print(f"P-value:         {p_value:.6f}")
    print("-"*60)

    if p_value < 0.05:
        print(f"--- Significant difference detected (p < 0.05).")
        print(f"   → {column_name} likely affects patient condition.")
    else:
        print(f"--- No significant difference detected (p ≥ 0.05).")
        print(f"   → {column_name} may not strongly differentiate groups.")
    print("="*60, "\n")