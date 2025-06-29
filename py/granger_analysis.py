import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

def check_stationarity_pro(df, adf_thresh=0.05):
    """
    DataFrame의 모든 열에 대해 ADF 검정을 수행하여 정상성 여부를 판단합니다.

    Args:
        df (pd.DataFrame): 검정할 데이터프레임.
        adf_thresh (float): 정상성 판단을 위한 p-value 임계값.

    Returns:
        dict: 각 컬럼을 Key로, 정상성 여부(True/False)를 Value로 갖는 딕셔너리.
    """
    stationarity_results = {}
    for col in df.columns:
        series = df[col].dropna()
        if len(series) < 20: # 데이터가 너무 적으면 검정 신뢰도 하락
            print(f"[Info] Not enough data for '{col}'. Assuming non-stationary.")
            stationarity_results[col] = False
            continue
        
        adf_result = adfuller(series, autolag='AIC')
        p_value = adf_result[1]
        stationarity_results[col] = p_value < adf_thresh
    return stationarity_results

def add_diff_columns_pro(
    df, 
    diff_cols, 
    diff_order=1, 
    transform='log', 
    suffix="_diff"
):
    """
    선택한 컬럼들에 대해 (로그)차분을 수행한 새 컬럼을 추가하고 결측치를 처리합니다.
    (이전 버전과 동일하며, 파이프라인의 일부로 사용됩니다)
    """
    df_new = df.copy()
    new_col_names = []

    for col in diff_cols:
        if col not in df_new.columns:
            print(f"[Warning] Column '{col}' not in DataFrame. Skipping.")
            continue
        
        new_col_name = col + suffix
        new_col_names.append(new_col_name)
        
        try:
            if transform == 'log':
                if (df_new[col] <= 0).any():
                    print(f"[Warning] Column '{col}' contains non-positive values. Log transform skipped; performing simple differencing instead.")
                    df_new[new_col_name] = df_new[col].diff(diff_order)
                else:
                    df_new[new_col_name] = np.log(df_new[col]).diff(diff_order)
            elif transform == 'simple':
                df_new[new_col_name] = df_new[col].diff(diff_order)
            else:
                print(f"[Warning] Invalid transform type '{transform}'. Skipping column '{col}'.")
                continue
        except Exception as e:
            print(f"[Error] Could not process column '{col}'. Error: {e}")
            
    df_processed = df_new.dropna(subset=new_col_names).reset_index(drop=True)
    return df_processed

def granger_causality_test_pro(
    df,
    base_vars,
    target_col,
    max_lag=5,
    p_thresh=0.05,
    verbose=False
):
    """
    주어진 변수들과 타겟 변수 간의 양방향 그랜저 인과관계 검정을 효율적으로 수행합니다.
    (이전 버전과 동일하며, 파이프라인의 일부로 사용됩니다)
    """
    results_list = []

    for var in base_vars:
        if var not in df.columns:
            print(f"[Warning] '{var}' column not in DataFrame. Skipping.")
            continue
        if target_col not in df.columns:
            raise ValueError(f"[Error] Target column '{target_col}' not in DataFrame.")

        df_tmp = df[[var, target_col]].dropna()
        if len(df_tmp) < 3 * max_lag:
            print(f"[Info] Not enough data for '{var}' vs '{target_col}'. Skipping.")
            continue
        
        directions = [
            {'causal': var, 'effect': target_col, 'data': df_tmp},
            {'causal': target_col, 'effect': var, 'data': df_tmp[[target_col, var]]}
        ]

        for direction in directions:
            causal_var, effect_var, test_data = direction['causal'], direction['effect'], direction['data']
            try:
                test_results = grangercausalitytests(test_data, maxlag=max_lag, verbose=verbose)
                for lag in range(1, max_lag + 1):
                    p_value = test_results[lag][0]['ssr_ftest'][1]
                    if p_value < p_thresh:
                        results_list.append({
                            "Causal_Var": causal_var,
                            "Effect_Var": effect_var,
                            "Lag": lag,
                            "P-Value": p_value
                        })
            except Exception as e:
                print(f"[Error] Granger test failed for {causal_var} -> {effect_var}. Error: {e}")

    if not results_list:
        return pd.DataFrame(columns=["Causal_Var", "Effect_Var", "Lag", "P-Value"])
        
    return pd.DataFrame(results_list).sort_values(by=["Causal_Var", "Effect_Var", "Lag"]).reset_index(drop=True)

def run_granger_analysis_pipeline(
    df,
    base_vars,
    target_col,
    max_lag=5,
    transform='log',
    suffix='_diff'
):
    """
    [NEW] 시계열 분석 파이프라인을 통합 실행하는 메인 함수입니다.
    1. 정상성(stationarity)을 자동으로 검사합니다.
    2. 비정상(non-stationary) 변수만 선택적으로 (로그)차분합니다.
    3. 최종적으로 정상화된 데이터로 그랜저 인과관계 검정을 수행합니다.
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임.
        base_vars (list): 분석할 기본 변수 리스트.
        target_col (str): 타겟 변수.
        max_lag (int): 그랜저 검정의 최대 래그.
        transform (str): 차분 방식 ('log' 또는 'simple').
    
    Returns:
        pd.DataFrame: 그랜저 인과관계 분석 결과.
    """
    print("--- 1. Starting Analysis Pipeline ---")
    all_vars = base_vars + [target_col]
    df_analysis = df[all_vars].copy()

    # --- 2. Check Stationarity ---
    print("\n--- 2. Checking variable stationarity (ADF Test) ---")
    stationarity = check_stationarity_pro(df_analysis)
    
    non_stationary_vars = [var for var, is_stationary in stationarity.items() if not is_stationary]
    stationary_vars = [var for var, is_stationary in stationarity.items() if is_stationary]
    
    print(f"Stationary variables: {stationary_vars or 'None'}")
    print(f"Non-stationary variables to be differenced: {non_stationary_vars or 'None'}")

    # --- 3. Apply Differencing Selectively ---
    if non_stationary_vars:
        print(f"\n--- 3. Applying '{transform}' differencing ---")
        df_processed = add_diff_columns_pro(df_analysis, non_stationary_vars, transform=transform, suffix=suffix)
    else:
        print("\n--- 3. No differencing needed. All variables are stationary. ---")
        df_processed = df_analysis.dropna() # Handle potential NaNs in original data
    
    # --- 4. Prepare Final Variables for Granger Test ---
    final_base_vars = []
    for var in base_vars:
        final_base_vars.append(var + suffix if var in non_stationary_vars else var)
        
    final_target_col = target_col + suffix if target_col in non_stationary_vars else target_col

    print("\n--- 4. Running Granger Causality Test on processed data ---")
    print(f"Base variables: {final_base_vars}")
    print(f"Target variable: {final_target_col}")

    # --- 5. Run Granger Causality Test ---
    report = granger_causality_test_pro(df_processed, final_base_vars, final_target_col, max_lag=max_lag)
    print("\n--- 5. Analysis Complete ---")
    return report


