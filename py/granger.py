def granger_causality_test_pro(
    df,
    base_vars,
    target_col="Exceedance",
    max_lag=5,
    p_thresh=0.05,
    verbose=False
):
    """
    주어진 변수들과 타겟 변수 간의 양방향 그랜저 인과관계 검정을 효율적으로 수행합니다.

    Args:
        df (pd.DataFrame): 시계열 데이터프레임
        base_vars (list): 검정을 수행할 변수 리스트
        target_col (str): 타겟 변수명
        max_lag (int): 테스트할 최대 래그
        p_thresh (float): 유의수준 임계값. 이 값보다 작은 p-value만 결과에 포함됩니다.
        verbose (bool): grangercausalitytests의 결과 출력 여부

    Returns:
        pd.DataFrame: 인과관계 검정 결과를 담은 데이터프레임
    """
    results_list = []

    for var in base_vars:
        # 1. 변수 존재 여부 및 데이터 유효성 검사
        if var not in df.columns:
            print(f"[Warning] '{var}' column not in DataFrame. Skipping.")
            continue
        if target_col not in df.columns:
            raise ValueError(f"[Error] Target column '{target_col}' not in DataFrame.")

        df_tmp = df[[var, target_col]].dropna()
        if len(df_tmp) < 3 * max_lag: # 안정적인 검정을 위해 충분한 데이터가 있는지 확인
            print(f"[Info] Not enough data for '{var}' vs '{target_col}'. Skipping.")
            continue

        # 2. 양방향 검정 수행 (효율적인 방식)
        # Direction 1: var -> target_col
        try:
            # grangercausalitytests를 한 번만 호출
            test_results_1 = grangercausalitytests(df_tmp, maxlag=max_lag, verbose=verbose)
            for lag in range(1, max_lag + 1):
                p_value = test_results_1[lag][0]['ssr_ftest'][1]
                if p_value < p_thresh:
                    results_list.append({
                        "Causal_Var": var,
                        "Effect_Var": target_col,
                        "Lag": lag,
                        "P-Value": p_value
                    })
        except Exception as e:
            print(f"[Error] Granger test failed for {var} -> {target_col}. Error: {e}")


        # Direction 2: target_col -> var
        try:
            # grangercausalitytests를 한 번만 호출 (열 순서 변경)
            test_results_2 = grangercausalitytests(df_tmp[[target_col, var]], maxlag=max_lag, verbose=verbose)
            for lag in range(1, max_lag + 1):
                p_value = test_results_2[lag][0]['ssr_ftest'][1]
                if p_value < p_thresh:
                    results_list.append({
                        "Causal_Var": target_col,
                        "Effect_Var": var,
                        "Lag": lag,
                        "P-Value": p_value
                    })
        except Exception as e:
            print(f"[Error] Granger test failed for {target_col} -> {var}. Error: {e}")

    # 3. 최종 결과를 DataFrame으로 변환
    if not results_list:
        return pd.DataFrame(columns=["Causal_Var", "Effect_Var", "Lag", "P-Value"])
        
    return pd.DataFrame(results_list).sort_values(by=["Causal_Var", "Effect_Var", "Lag"])
