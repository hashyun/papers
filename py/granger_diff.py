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
    target_col,
    max_lag=5,
    p_thresh=0.05
):
    """
    선택된 base_vars에 대해 Granger 인과관계 테스트를 양방향으로 수행.
    df: 입력 데이터프레임
    base_vars: Granger 테스트 대상 변수들 (문자열 리스트)
    target_col: 타깃 변수 (문자열)
    max_lag: 테스트할 최대 lag
    p_thresh: 유의수준 기준 (예: 0.05)
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
                # 테스트: var -> target_col
                test_result_1 = grangercausalitytests(df_tmp, maxlag=lag)
                p_value_1 = test_result_1[lag][0]['ssr_ftest'][1]
                result_dict[var]["var_to_target"][lag] = p_value_1

                # 테스트: target_col -> var (열 순서 변경)
                df_tmp_rev = df_tmp[[target_col, var]]
                test_result_2 = grangercausalitytests(df_tmp_rev, maxlag=lag)
                p_value_2 = test_result_2[lag][0]['ssr_ftest'][1]
                result_dict[var]["target_to_var"][lag] = p_value_2

            except Exception as e:
                print(f"[Warning] Granger test fail: var={var}, lag={lag}, error={e}")
                continue

    return result_dict
