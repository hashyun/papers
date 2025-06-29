import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

def add_diff_columns(df, diff_cols, diff_order=1, suffix="_diff"):
    """
    선택한 컬럼(diff_cols)들에 대해 차분 열을 추가하고, 해당 차분 열에 NaN이 있는 행은 제거한 DataFrame 반환.
    이 방식은 차분에 의해 발생하는 결측치 때문에 데이터프레임 크기가 달라지는 문제를 해결합니다.
    """
    df_new = df.copy()
    for col in diff_cols:
        if col in df_new.columns:
            df_new[col + suffix] = np.log(df[col]).diff(diff_order)
        else:
            print(f"[Warning] {col} not in DataFrame. Skipping differencing for this column.")
    # 차분된 컬럼들에 NaN이 포함된 행을 제거 (모든 차분 변수에서 결측치가 없는 인덱스만 사용)
    diff_cols_new = [col + suffix for col in diff_cols if col in df_new.columns]
    df_new = df_new.dropna(subset=diff_cols_new)
    return df_new

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
                test_result_1 = grangercausalitytests(df_tmp, maxlag=lag, verbose=False)
                p_value_1 = test_result_1[lag][0]['ssr_ftest'][1]
                result_dict[var]["var_to_target"][lag] = p_value_1

                # target_col -> var (반대 방향)
                df_tmp_rev = df_tmp[[target_col, var]]  # 열 순서 바꿈
                test_result_2 = grangercausalitytests(df_tmp_rev, maxlag=lag, verbose=False)
                p_value_2 = test_result_2[lag][0]['ssr_ftest'][1]
                result_dict[var]["target_to_var"][lag] = p_value_2

            except Exception as e:
                print(f"[Warning] Granger test fail: var={var}, lag={lag}, error={e}")
                continue

    return result_dict
