
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

def granger_causality_test_bidirectional(
    df,
    base_vars,
    target_col="Exceedance",
    max_lag=5,
    p_thresh=0.05
):
    """
    (1) base_vars에 있는 '원본 변수'들에 대해 Granger test 실행
    (2) 각 var에 대해 lag=1..max_lag 반복
       - p-value < p_thresh이면 그 lag가 유의한 것으로 판단
    (3) 양방향( var -> target_col, target_col -> var ) 테스트 수행
    (4) 결과 { var: { 'var_to_target': {lag1: p, lag2: p, ...}, 'target_to_var': {lag1: p, lag2: p, ...} } } 반환
    """
    result_dict = {}
    for var in base_vars:
        if var not in df.columns:
            print(f"[Warning] {var} not in df => skip.")
            continue
        if target_col not in df.columns:
            raise ValueError(f"[Error] {target_col} not in df.columns.")

        df_tmp = df[[var, target_col]].dropna()
        if df_tmp.shape[0] < (max_lag + 1):
            print(f"[Info] Not enough data for {var} => skip.")
            continue

        result_dict[var] = {"var_to_target": {}, "target_to_var": {}}

        for lag in range(1, max_lag + 1):
            try:
                # var -> target_col
                test_result_1 = grangercausalitytests(df_tmp, maxlag=lag)
                p_value_1 = test_result_1[lag][0]['ssr_ftest'][1]
                result_dict[var]["var_to_target"][lag] = p_value_1

                # target_col -> var (반대 방향)
                df_tmp_rev = df_tmp[[target_col, var]]  # 열 순서 바꿈
                test_result_2 = grangercausalitytests(df_tmp_rev, maxlag=lag)
                p_value_2 = test_result_2[lag][0]['ssr_ftest'][1]
                result_dict[var]["target_to_var"][lag] = p_value_2

            except Exception as e:
                print(f"[Warning] Granger test fail: var={var}, lag={lag}, error={e}")
                continue

    return result_dict

