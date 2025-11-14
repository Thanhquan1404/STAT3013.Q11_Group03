import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def correlation_test(df: pd.DataFrame, column_label: str, threshold: float = 0.8):
    print("="*30 + " Analyzing correlation matrix " + "="*30)

    corr_matrix = df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.show()

    if column_label in corr_matrix.columns:
        print("\n---Correlation of each feature with target variable:")
        target_corr = corr_matrix[column_label].sort_values(ascending=False)
        print(target_corr)

        print("\n---Analysis:")
        print("- Features with correlation close to ±1 are highly predictive (may dominate results).")
        print("- Features with near-zero correlation may not help much.")
        print("- Very high correlation (> 0.8) between two features may cause multicollinearity.")
    else:
        print(f"Target column '{column_label}' not found in the correlation matrix!")

    high_corr_pairs = []
    for col in corr_matrix.columns:
        for row in corr_matrix.columns:
            if col != row and abs(corr_matrix.loc[row, col]) > threshold:
                high_corr_pairs.append((row, col, corr_matrix.loc[row, col]))

    if high_corr_pairs:
        print("\nStrongly correlated feature pairs (|corr| > {:.2f}):".format(threshold))
        for pair in high_corr_pairs:
            print(f"   {pair[0]} ↔ {pair[1]}  | corr = {pair[2]:.3f}")
    else:
        print("\nNo highly correlated feature pairs found above the threshold.")

    print("="*80)