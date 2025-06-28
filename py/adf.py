
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def check_stationarity(df, columns):
    """
    Augmented Dickey-Fuller (ADF) Test를 사용하여 각 변수의 정상성을 효율적으로 검정
    - 메모리 절약을 위해 결과만 저장하고, 출력은 별도 수행
    """
    adf_results = []
    for col in columns:
        if col not in df.columns:
            print(f"[Warning] {col} not in df => skip.")
            continue
        series = df[col].dropna().values  # Pandas Series -> NumPy 배열로 변환하여 메모리 절약
        result = adfuller(series, autolag=None,maxlag=5)  # 자동 래그 선택 (AIC 기준)
        adf_results.append({
            "Variable": col,
            "ADF Statistic": result[0],
            "p-value": result[1],
            "Critical Values": result[4]
        })

    return pd.DataFrame(adf_results)  # 결과를 DataFrame으로 변환하여 효율적 출력
