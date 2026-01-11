from scipy.stats import pearsonr

def analyze_correlations(df):
    """
    Performs statistical analysis.
    Uses try-except to handle cases where columns might be missing or non-numeric.
    """
    print("\n--- Starting Statistical Analysis ---")

    try:
        # Verify required columns exist
        if 'DietQuality' not in df.columns or 'MMSE' not in df.columns:
            raise ValueError("Missing columns: 'DietQuality' or 'MMSE' not found.")

        # Attempt calculation
        corr, p_value = pearsonr(df['DietQuality'], df['MMSE'])
        
        print(f"Diet vs MMSE Correlation: {corr:.4f}")
        print(f"P-Value: {p_value:.4f}")

        if p_value < 0.05:
            print(">> Result is Statistically Significant.")
        else:
            print(">> Result is NOT Statistically Significant.")

    except Exception as e:
        print(f"   [Error] Statistical analysis failed: {e}")
    
    print("-------------------------------------\n")