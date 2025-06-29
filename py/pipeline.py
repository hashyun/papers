import pandas as pd
import numpy as np

# Import functions from your scripts
from adf import check_stationarity
from granger_analysis import run_granger_analysis_pipeline
from hill_plot_new import select_threshold
import l1_l2_gpd_new # Import the whole module

def main():
    """
    Main function to run the entire data analysis pipeline.
    """
    # --- 1. Load Data ---
    print("--- 1. Loading Data ---")
    n_samples = 2000
    rng = np.random.default_rng(42)
    data = {
        'feature1': np.random.randn(n_samples).cumsum() + 50,
        'feature2': np.random.randn(n_samples).cumsum() + 20,
        'target': np.random.randn(n_samples)
    }
    # Introduce some extreme values in the target
    extreme_indices = rng.choice(n_samples, size=100, replace=False)
    from scipy import stats
    data['target'][extreme_indices] = stats.genpareto.rvs(c=0.2, scale=5, size=100, random_state=rng) + 5
    df = pd.DataFrame(data)
    print("Sample data created.")
    print(df.head())

    # Define variables for the analysis
    ALL_VARS = ['feature1', 'feature2', 'target']
    TARGET_COL = 'target'
    BASE_VARS = ['feature1', 'feature2']
    FEATURE_COLS = ['feature1', 'feature2'] # For the final model

    # --- 2. ADF Test for Stationarity ---
    print()
    print("--- 2. Running ADF Test ---")
    adf_results = check_stationarity(df, columns=ALL_VARS)
    print("ADF Test Results:")
    print(adf_results)

    # --- 3. Granger Causality Analysis ---
    print()
    print("--- 3. Running Granger Causality Analysis ---")
    # The granger analysis pipeline handles differencing internally
    granger_results = run_granger_analysis_pipeline(
        df=df.copy(),
        base_vars=BASE_VARS,
        target_col=TARGET_COL,
        max_lag=5
    )
    print("Granger Causality Results:")
    print(granger_results)
    # In a real-world scenario, you might use these results to select your features.
    # For this pipeline, we'll proceed with the predefined FEATURE_COLS.

    # --- 4. Hill Plot for Threshold Selection ---
    print()
    print(f"--- 4. Running Hill Plot on Target Column: '{TARGET_COL}' ---")
    # We use the original, non-differenced target variable for threshold selection.
    target_series = df[TARGET_COL].dropna().values
    
    # The select_threshold function returns: u_hat, k_hat, gamma_hat
    threshold, _, _ = select_threshold(target_series)
    print(f"Threshold selected via Hill Plot: {threshold:.4f}")

    # --- 5. L2 / GPD Modeling ---
    print()
    print("--- 5. Running L2-CART and GPD-CART Models ---")
    # This is the final step, using the original data and the calculated threshold.
    pipeline_results = l1_l2_gpd_new.run_l1_l2_gpd_pipeline(
        df=df.copy(),
        target_col=TARGET_COL,
        feature_cols=FEATURE_COLS,
        threshold=threshold,
        lags=[1, 2, 5],
        max_depth=4,
        min_leaf_l2=30,
        min_leaf_gpd=30,
        cv_folds_gpd=5,
        run_l2=True,
        run_gpd=True
    )

    print()
    print("--- Pipeline Finished ---")
    print("Final results dictionary contains the trained models and other info.")
    # You can inspect the results, for example, the pruned GPD tree:
    if pipeline_results and pipeline_results.get('gpd_tree_pruned'):
        print()
        print("--- Final Pruned GPD Tree Structure ---")
        l1_l2_gpd_new.print_tree_structure(pipeline_results['gpd_tree_pruned'], pipeline_results['lagged_feature_names'])
    else:
        print()
        print("Could not generate a pruned GPD tree. The tail may have had insufficient data after splitting.")


if __name__ == '__main__':
    main()
