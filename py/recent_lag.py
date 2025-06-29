def get_most_recent_granger_lags(granger_df: pd.DataFrame, target_variable: str = 'Exceedance') -> pd.DataFrame:
    """
    그랜저 인과관계 분석 결과에서 각 원인 변수가 타겟 변수에 영향을 미치는
    가장 빠른(가장 작은) 시차(Lag)를 추출합니다.

    Args:
        granger_df (pd.DataFrame): granger_causality_test_pro 함수의 결과 데이터프레임.
                                   ['Causal_Var', 'Effect_Var', 'Lag', 'P-Value'] 컬럼 필요.
        target_variable (str): 분석의 대상이 되는 결과 변수(Effect_Var)의 이름.

    Returns:
        pd.DataFrame: 각 원인 변수와 그에 해당하는 가장 빠른 시차 정보를 담은 데이터프레임.
    """
    # 1. 입력 데이터프레임 유효성 검사
    required_cols = ['Causal_Var', 'Effect_Var', 'Lag']
    if not all(col in granger_df.columns for col in required_cols):
        raise ValueError(f"입력 데이터프레임에 필수 컬럼이 없습니다: {required_cols}")

    if granger_df.empty:
        print("입력 데이터프레임이 비어있습니다.")
        return pd.DataFrame(columns=['Causal_Var', 'Most_Recent_Lag'])

    # 2. 결과 변수(Effect_Var)가 타겟 변수인 경우만 필터링
    df_filtered = granger_df[granger_df['Effect_Var'] == target_variable].copy()

    if df_filtered.empty:
        print(f"'{target_variable}'에 영향을 미치는 유의미한 변수를 찾지 못했습니다.")
        return pd.DataFrame(columns=['Causal_Var', 'Most_Recent_Lag'])

    # 3. 'Lag' 값을 기준으로 오름차순 정렬
    df_sorted = df_filtered.sort_values(by='Lag', ascending=True)

    # 4. 'Causal_Var'별로 중복을 제거하고 가장 첫 번째(가장 작은 Lag) 값만 남김
    df_result = df_sorted.drop_duplicates(subset=['Causal_Var'], keep='first')

    # 5. 결과 컬럼 선택 및 이름 변경
    df_result = df_result[['Causal_Var', 'Lag']].rename(columns={'Lag': 'Most_Recent_Lag'})
    
    # 인덱스 리셋
    df_result = df_result.reset_index(drop=True)

    return df_result
