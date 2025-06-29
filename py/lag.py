import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Optional, Dict

# 로깅 설정 (함수 테스트를 위해 포함)
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def create_lagged_features(
    df: pd.DataFrame,
    target_col: str,
    lag_config: Dict[str, List[int]],
    date_col: str,
    start_year: int,
    additional_features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    시계열 데이터에 대해 시차(lag) 및 차분(difference) 변수를 생성합니다.

    - 안정성 개선: 로그 차분 시, 0 또는 음수 값이 존재할 경우 오류 대신 단순 차분으로 자동 전환합니다.
    - 명확성 증대: 데이터프레임 결합 로직을 더 명시적으로 변경하고, 입력값 검증을 추가했습니다.

    Args:
        df (pd.DataFrame): 원본 시계열 데이터프레임.
        target_col (str): 모델의 타겟 변수 컬럼명.
        lag_config (Dict[str, List[int]]): {'변수명': [시차1, 시차2...]} 형식의 딕셔너리.
                                            변수명이 '_diff'로 끝나면 로그 차분 후 시차 적용.
        date_col (str): 날짜 정보가 있는 컬럼명.
        start_year (int): 분석을 시작할 연도. 이 연도 이전의 데이터는 필터링됩니다.
        additional_features (Optional[List[str]], optional): 시차 없이 포함할 추가적인 피처 컬럼명 리스트.

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            - 최종적으로 생성된 피처와 타겟을 포함한 데이터프레임.
            - 모델에 사용된 피처 이름 리스트.
    """
    df_copy = df.copy()

    # 1. 입력 컬럼 존재 여부 확인
    all_vars_needed = {target_col} | set(lag_config.keys()) | set(additional_features or [])
    for var in all_vars_needed:
        base_var = var[:-5] if var.endswith('_diff') else var
        if base_var not in df_copy.columns:
            raise ValueError(f"필수 컬럼 '{base_var}'가 데이터프레임에 존재하지 않습니다.")

    # 2. 날짜 기준 정렬 및 연도 컬럼 생성
    try:
        if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        df_copy['_year_'] = df_copy[date_col].dt.year
    except Exception as e:
        raise ValueError(f"날짜 컬럼 '{date_col}' 처리 중 오류 발생: {e}")

    df_copy = df_copy.sort_values(by=date_col).reset_index(drop=True)

    # 3. 시차 변수 생성
    features_to_concat = []
    for var, lags in lag_config.items():
        if var.endswith("_diff"):
            base_var = var[:-5]
            # 안정성: 로그 변환 전 0 또는 음수 값 확인
            if (df_copy[base_var] <= 0).any():
                logging.warning(f"'{base_var}' 컬럼에 0 이하의 값이 있어 로그 차분 대신 단순 차분을 적용합니다.")
                series_to_lag = df_copy[base_var].diff(1)
            else:
                series_to_lag = np.log(df_copy[base_var]).diff(1)
            
            for lag in lags:
                if lag > 0:
                    lagged_col = series_to_lag.shift(lag)
                    lagged_col.name = f"{var}_lag{lag}"
                    features_to_concat.append(lagged_col)
        else:
            for lag in lags:
                if lag > 0:
                    lagged_col = df_copy[var].shift(lag)
                    lagged_col.name = f"{var}_lag{lag}"
                    features_to_concat.append(lagged_col)

    # 4. 데이터 결합
    model_df = pd.concat(features_to_concat, axis=1)

    # 원본 데이터에서 타겟 및 추가 피처 컬럼들을 결합
    cols_to_join = [target_col, '_year_'] + (additional_features or [])
    model_df = model_df.join(df_copy[cols_to_join])

    # 5. 최종 데이터프레임 정제
    if start_year is not None:
        model_df = model_df[model_df['_year_'] >= start_year]

    # 시차 생성으로 발생한 NaN 값을 가진 행 제거
    final_df = model_df.dropna().reset_index(drop=True)
    
    # 임시 연도 컬럼 제거
    if '_year_' in final_df.columns:
        final_df = final_df.drop(columns=['_year_'])

    # 최종적으로 사용된 피처 이름 목록 추출
    feature_names = [col for col in final_df.columns if col != target_col]

    logging.info(f"시차 변수 생성 완료. 최종 데이터 크기: {len(final_df)} 샘플, {len(feature_names)} 피처.")

    return final_df, feature_names

# --- 예시 사용법 ---
# if __name__ == '__main__':
#     # 가상 데이터 생성
#     dates = pd.to_datetime(pd.date_range(start='2010-01-01', periods=500, freq='D'))
#     data = {
#         'Date': dates,
#         'Price': 100 + np.random.randn(500).cumsum(),
#         'Volume': 1000 + np.random.randint(-50, 50, 500).cumsum(),
#         'Category': ['A'] * 250 + ['B'] * 250,
#         'Target': np.random.rand(500) * 10
#     }
#     sample_df = pd.DataFrame(data)
#     sample_df.loc[10, 'Price'] = 0 # 로그 변환 테스트를 위한 0 값 삽입

#     # 시차 설정
#     lag_conf = {
#         'Price_diff': [1, 2, 7],
#         'Volume': [1, 7]
#     }

#     # 함수 실행
#     model_ready_df, features = create_lagged_features(
#         df=sample_df,
#         target_col='Target',
#         lag_config=lag_conf,
#         date_col='Date',
#         start_year=2011,
#         additional_features=['Category']
#     )

#     print("--- 최종 데이터프레임 ---")
#     print(model_ready_df.head())
#     print("\n--- 사용된 피처 목록 ---")
#     print(features)
#     print(f"\n데이터 형태: {model_ready_df.shape}")

