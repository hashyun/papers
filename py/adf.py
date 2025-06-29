
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def check_stationarity(df, columns, significance_level=0.05):
    """
    Augmented Dickey-Fuller (ADF) Test를 사용하여 각 변수의 정상성을 효율적으로 검정합니다.
    - 결과를 DataFrame으로 반환하여 가독성과 활용성을 높입니다.
    - p-value를 기준으로 정상성 여부를 판단하는 'Is Stationary' 컬럼을 추가합니다.
    """
    adf_results = []
    for col in columns:
        if col not in df.columns:
            print(f"[Warning] '{col}' column not in DataFrame. Skipping.")
            continue
            
        # .dropna()로 결측치 제거 후 NumPy 배열로 변환하여 메모리 효율성 증대
        series = df[col].dropna().values
        
        # 만약 dropna 후 데이터가 거의 없다면 테스트를 건너뜀
        if len(series) < 2 * 5: # maxlag의 두 배보다 적으면 불안정할 수 있음
             print(f"[Warning] Not enough data for '{col}' after dropping NA. Skipping.")
             continue

        # autolag='AIC'로 설정하여 AIC를 기준으로 최적의 래그를 자동으로 선택
        result = adfuller(series, autolag='AIC', maxlag=5)
        
        adf_results.append({
            "Variable": col,
            "ADF Statistic": result[0],
            "p-value": result[1],
            "Lags Used": result[2], # 사용된 래그 수도 함께 보면 좋음
            "Is Stationary (p < " + str(significance_level) + ")": result[1] < significance_level,
            "Critical Value (5%)": result[4]['5%'] # 5% 임계값을 바로 비교할 수 있도록 추출
        })

    return pd.DataFrame(adf_results)
