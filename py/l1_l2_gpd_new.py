import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize
from sklearn.model_selection import KFold, StratifiedKFold
import copy
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from scipy import stats
import math

# =====================================================================
# 1. 로깅 및 상수 설정
# =====================================================================
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

@dataclass(frozen=True)
class Constants:
    SEED: int = 42
    MIN_FLOAT: float = np.finfo(float).eps
    LARGE_FLOAT: float = 1e30
    GPD_GAMMA_BOUNDS: Tuple[float, float] = (-0.5, 0.95)
    GPD_MIN_LEAF_DEFAULT: int = 30
    AIC_PENALTY: float = 2.0

C = Constants()
rng = np.random.default_rng(C.SEED)

# =====================================================================
# 2. Node 클래스 정의
# =====================================================================
@dataclass
class NodeBase:
    is_leaf: bool = True
    depth: int = 0
    n_samples: int = 0
    split_var: Optional[int] = None
    split_thr: Optional[float] = None
    left: Optional['NodeBase'] = None
    right: Optional['NodeBase'] = None

@dataclass
class NodeL1(NodeBase):
    median_val: Optional[float] = None
    gain: float = 0.0
    lognorm_mu: Optional[float] = None
    lognorm_sigma: Optional[float] = None

@dataclass
class NodeL2(NodeBase):
    mean_val: Optional[float] = None
    gain: float = 0.0
    lognorm_mu: Optional[float] = None
    lognorm_sigma: Optional[float] = None

@dataclass
class NodeGPD(NodeBase):
    gpd_params: Optional[Tuple[float, float]] = None
    nll: float = C.LARGE_FLOAT
    split_gain: float = 0.0

# =====================================================================
# 3. 핵심 함수들
# =====================================================================

def create_lagged_features(
    df: pd.DataFrame,
    target_col: str,
    lag_config: Dict[str, List[int]],
    date_col: str,
    start_year: int,
    additional_features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    설정(config)에 따라 시차 및 차분 변수를 생성합니다.
    (오류가 발생하지 않도록 수정 및 안정화된 버전)
    """
    logging.info("시차 변수 생성 중...")
    df_copy = df.copy()

    try:
        if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    except Exception as e:
        raise ValueError(f"'{date_col}' 컬럼을 datetime으로 변환할 수 없습니다: {e}")

    df_copy = df_copy.sort_values(by=date_col).reset_index(drop=True)
    
    features_list = []
    
    for var, lags in lag_config.items():
        series_to_lag = None
        # 차분 변수 처리
        if var.endswith("_diff"):
            base_var = var.replace("_diff", "")
            if base_var not in df_copy.columns:
                logging.warning(f"차분을 위한 원본 컬럼 '{base_var}'를 찾을 수 없습니다. '{var}'를 건너뜁니다.")
                continue
            
            series = df_copy[base_var]
            # 안정성: 로그 변환 전 0 또는 음수 값 확인
            if (series <= 0).any():
                logging.warning(f"'{base_var}' 컬럼에 0 이하의 값이 있어 단순 차분을 적용합니다.")
                series_to_lag = series.diff()
            else:
                series_to_lag = np.log(series).diff()
        # 일반 시차 변수 처리
        else:
            if var not in df_copy.columns:
                logging.warning(f"시차를 적용할 '{var}' 컬럼을 찾을 수 없습니다. 건너뜁니다.")
                continue
            series_to_lag = df_copy[var]

        for lag in lags:
            if lag > 0:
                new_col_name = f"{var}_lag{lag}"
                shifted_series = series_to_lag.shift(lag)
                shifted_series.name = new_col_name
                features_list.append(shifted_series)

    # 생성된 시차 변수들을 하나의 데이터프레임으로 결합
    model_df = pd.concat(features_list, axis=1) if features_list else pd.DataFrame(index=df_copy.index)

    # 원본 데이터에서 타겟 및 추가 피처 컬럼들을 결합
    cols_to_join = [target_col] + (additional_features or [])
    model_df = model_df.join(df_copy[cols_to_join + [date_col]])
    
    # 시작 연도 필터링 및 NaN 제거
    if start_year is not None:
        model_df = model_df[model_df[date_col].dt.year >= start_year].copy()
    
    final_df = model_df.dropna().reset_index(drop=True)
    
    feature_names_used = [col for col in final_df.columns if col not in [target_col, date_col]]
    
    logging.info(f"피처 생성 완료. 최종 데이터 크기: {len(final_df)} 샘플, {len(feature_names_used)} 피처.")
    
    return final_df, feature_names_used

# --- 손실 함수 ---
def mae_loss(y: np.ndarray, md: float) -> float:
    return np.sum(np.abs(y - md)) if len(y) > 0 else 0.0

def mse_loss(y: np.ndarray, mu: float) -> float:
    return np.sum((y - mu) ** 2) if len(y) > 0 else 0.0

# --- 분포 피팅 함수 ---
def _nll_lognormal(params: Tuple[float, float], y: np.ndarray, *, trunc_left: float = 0.0) -> float:
    mu, sigma = params
    if sigma <= C.MIN_FLOAT: return C.LARGE_FLOAT
    y_pos = y[y > 0]
    if y_pos.size == 0: return 0.0 if trunc_left > 0 else C.LARGE_FLOAT
    log_y = np.log(y_pos)
    z = (log_y - mu) / sigma
    base_nll = np.sum(np.log(sigma) + 0.5 * z**2 + log_y) + 0.5 * y_pos.size * np.log(2 * np.pi)
    if trunc_left > 0:
        zc = (np.log(trunc_left) - mu) / sigma
        surv = stats.norm.sf(zc)
        if surv <= C.MIN_FLOAT: return C.LARGE_FLOAT
        base_nll -= y_pos.size * np.log(surv)
    return base_nll if np.isfinite(base_nll) else C.LARGE_FLOAT

def fit_lognormal_mle(y: np.ndarray, *, trunc_left: float = 0.0) -> Tuple[float, float]:
    y_pos = y[y > 0]
    if y_pos.size < 2: return (np.log(y_pos[0]), 1.0) if y_pos.size == 1 else (0.0, 1.0)
    log_y = np.log(y_pos)
    init = [log_y.mean(), max(log_y.std(ddof=1), C.MIN_FLOAT * 10)]
    bounds = [(None, None), (C.MIN_FLOAT * 10, None)]
    try:
        res = minimize(lambda p: _nll_lognormal(p, y_pos, trunc_left=trunc_left), init, method="L-BFGS-B", bounds=bounds)
        if res.success: return float(res.x[0]), float(max(res.x[1], C.MIN_FLOAT * 10))
    except Exception as e:
        logging.debug(f"LogNormal MLE failed: {e}")
    return tuple(init)

def _neg_loglik_gpd(theta: Tuple[float, float], y: np.ndarray) -> float:
    sigma, gamma = theta
    if sigma <= C.MIN_FLOAT or len(y) == 0: return 0.0 if len(y) == 0 else C.LARGE_FLOAT
    if abs(gamma) < 1e-6:
        if np.any(y < 0): return C.LARGE_FLOAT
        nll = len(y) * np.log(sigma) + np.sum(y) / sigma
    else:
        z = 1 + gamma * y / sigma
        if np.any(z <= C.MIN_FLOAT): return C.LARGE_FLOAT
        nll = len(y) * np.log(sigma) + (1 / gamma + 1) * np.sum(np.log(z))
    return nll if np.isfinite(nll) else C.LARGE_FLOAT

def fit_gpd_mle(y: np.ndarray) -> Tuple[float, float]:
    n = len(y)
    if n < C.GPD_MIN_LEAF_DEFAULT:
        med = np.median(y) if n > 0 else 1.0
        return max(med, C.MIN_FLOAT), -0.1
    try:
        y_sorted = np.sort(y)
        b0, b1 = np.mean(y_sorted), np.mean(y_sorted * (np.arange(n) / (n - 1)))
        l1, l2 = b0, 2 * b1 - b0
        tau2 = l2 / l1 if abs(l1) > C.MIN_FLOAT else 0
        gamma0 = (2 * tau2 - 1) / (tau2 - 1) if abs(tau2 - 1) > C.MIN_FLOAT else 0
        sigma0 = l1 * (1 - gamma0)
        gamma0 = np.clip(gamma0, C.GPD_GAMMA_BOUNDS[0] + 1e-4, C.GPD_GAMMA_BOUNDS[1] - 1e-4)
        sigma0 = max(sigma0, C.MIN_FLOAT * 10)
        init_params = (sigma0, gamma0)
    except Exception:
        init_params = (np.mean(y), 0.1)
    try:
        res = minimize(_neg_loglik_gpd, init_params, args=(y,), method="L-BFGS-B", bounds=[(C.MIN_FLOAT * 10, None), C.GPD_GAMMA_BOUNDS])
        if res.success:
            return float(res.x[0]), float(res.x[1])
    except Exception as e:
        logging.debug(f"GPD MLE 최적화 실패: {e}")
    return init_params

def node_cost_gpd(y: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    if len(y) < C.GPD_MIN_LEAF_DEFAULT: return C.LARGE_FLOAT, (1.0, 0.0)
    params = fit_gpd_mle(y)
    return _neg_loglik_gpd(params, y), params

# --- 분기 함수 ---
def _best_split_l1(X_col: np.ndarray, y: np.ndarray, min_leaf: int) -> Tuple[float, Optional[float]]:
    n = len(y)
    if n < 2 * min_leaf: return 0.0, None
    idx = np.argsort(X_col)
    xs, ys = X_col[idx], y[idx]
    parent_loss = mae_loss(ys, np.median(ys))
    best_gain, best_thr = 0.0, None
    for i in range(min_leaf, n - min_leaf + 1):
        if i < len(xs) and xs[i-1] == xs[i]: continue
        left_y, right_y = ys[:i], ys[i:]
        child_loss = mae_loss(left_y, np.median(left_y)) + mae_loss(right_y, np.median(right_y))
        gain = parent_loss - child_loss
        if gain > best_gain:
            best_gain, best_thr = gain, (xs[i-1] + xs[i]) / 2.0
    return (best_gain, best_thr) if best_gain > C.MIN_FLOAT else (0.0, None)

def _best_split_l2(X_col: np.ndarray, y: np.ndarray, min_leaf: int) -> Tuple[float, Optional[float]]:
    n = len(y)
    if n < 2 * min_leaf: return 0.0, None
    idx = np.argsort(X_col)
    xs, ys = X_col[idx], y[idx]
    parent_loss = mse_loss(ys, np.mean(ys))
    csum_y, csum_y2 = np.cumsum(ys), np.cumsum(ys**2)
    best_gain, best_thr = 0.0, None
    for i in range(min_leaf, n - min_leaf + 1):
        if i < len(xs) and xs[i-1] == xs[i]: continue
        sum_l, sum_sq_l = csum_y[i-1], csum_y2[i-1]
        sum_r, sum_sq_r = csum_y[-1] - sum_l, csum_y2[-1] - sum_sq_l
        cnt_l, cnt_r = i, n - i
        loss_l = sum_sq_l - (sum_l**2) / cnt_l
        loss_r = sum_sq_r - (sum_r**2) / cnt_r if cnt_r > 0 else 0
        gain = parent_loss - (loss_l + loss_r)
        if gain > best_gain:
            best_gain, best_thr = gain, (xs[i-1] + xs[i]) / 2.0
    return (best_gain, best_thr) if best_gain > C.MIN_FLOAT else (0.0, None)
    
def _best_split_gpd(X_col: np.ndarray, y: np.ndarray, min_leaf: int) -> Tuple[float, Optional[float]]:
    n = len(y)
    if n < 2 * min_leaf: return 0.0, None
    idx = np.argsort(X_col)
    xs, ys = X_col[idx], y[idx]
    parent_nll, _ = node_cost_gpd(ys)
    if not np.isfinite(parent_nll): return 0.0, None
    best_gain, best_thr = -C.LARGE_FLOAT, None
    for i in range(min_leaf, n - min_leaf + 1):
        if i < len(xs) and xs[i-1] == xs[i]: continue
        l_nll, _ = node_cost_gpd(ys[:i])
        r_nll, _ = node_cost_gpd(ys[i:])
        if not (np.isfinite(l_nll) and np.isfinite(r_nll)): continue
        gain = parent_nll - (l_nll + r_nll) - C.AIC_PENALTY
        if gain > best_gain:
            best_gain, best_thr = gain, (xs[i-1] + xs[i]) / 2.0
    return (best_gain, best_thr) if best_gain > 0 else (0.0, None)

# --- 트리 성장 함수 ---
def _grow_tree(X: np.ndarray, y: np.ndarray, node_type: type, split_finder: callable, feature_names: List[str], min_leaf: int, max_depth: int, depth: int = 0) -> NodeBase:
    n_samples, n_features = X.shape
    node = node_type()
    node.depth, node.n_samples = depth, n_samples
    if node_type == NodeL1: node.median_val = np.median(y) if n_samples > 0 else 0.0
    elif node_type == NodeL2: node.mean_val = np.mean(y) if n_samples > 0 else 0.0
    elif node_type == NodeGPD: node.nll, node.gpd_params = node_cost_gpd(y)
    is_stoppable = (depth >= max_depth or n_samples < 2 * min_leaf or (node_type == NodeGPD and not np.isfinite(node.nll)))
    if is_stoppable: return node
    best_gain, best_var, best_thr = (0.0, None, None) if node_type != NodeGPD else (-C.LARGE_FLOAT, None, None)
    for j in range(n_features):
        gain, thr = split_finder(X[:, j], y, min_leaf)
        if thr is not None and gain > best_gain:
            best_gain, best_var, best_thr = gain, j, thr
    if best_thr is None: return node
    mask_left = X[:, best_var] <= best_thr
    if np.sum(mask_left) < min_leaf or np.sum(~mask_left) < min_leaf: return node
    node.is_leaf = False
    node.split_var, node.split_thr = best_var, best_thr
    if isinstance(node, (NodeL1, NodeL2)): node.gain = best_gain
    elif isinstance(node, NodeGPD): node.split_gain = best_gain
    logging.debug(f"Depth {depth}: Split on '{feature_names[best_var]}' <= {best_thr:.3g}, Gain={best_gain:.3f}")
    node.left = _grow_tree(X[mask_left], y[mask_left], node_type, split_finder, feature_names, min_leaf, max_depth, depth + 1)
    node.right = _grow_tree(X[~mask_left], y[~mask_left], node_type, split_finder, feature_names, min_leaf, max_depth, depth + 1)
    return node

def grow_tree_l1(X: np.ndarray, y: np.ndarray, **kwargs) -> NodeL1:
    return _grow_tree(X, y, NodeL1, _best_split_l1, depth=0, **kwargs)

def grow_tree_l2(X: np.ndarray, y: np.ndarray, **kwargs) -> NodeL2:
    return _grow_tree(X, y, NodeL2, _best_split_l2, depth=0, **kwargs)

def grow_tree_gpd(X: np.ndarray, y: np.ndarray, **kwargs) -> NodeGPD:
    kwargs.setdefault('min_leaf', C.GPD_MIN_LEAF_DEFAULT)
    return _grow_tree(X, y, NodeGPD, _best_split_gpd, depth=0, **kwargs)

# --- 트리 탐색, 파라미터 할당, 가지치기 및 시각화 ---
def find_leaf(node: NodeBase, x: np.ndarray) -> NodeBase:
    while not node.is_leaf:
        node = node.left if x[node.split_var] <= node.split_thr else node.right
    return node

def assign_lognorm_params(root: Union[NodeL1, NodeL2], X: np.ndarray, y: np.ndarray, *, trunc_left: float = 0.0) -> None:
    if not root: return
    leaf_assignments = [find_leaf(root, x_i) for x_i in X]
    unique_leaves = list(set(leaf_assignments))
    for leaf in unique_leaves:
        mask = [id(la) == id(leaf) for la in leaf_assignments]
        y_leaf = y[mask]
        if y_leaf.size > 0:
            leaf.lognorm_mu, leaf.lognorm_sigma = fit_lognormal_mle(y_leaf, trunc_left=trunc_left)
        else:
            leaf.lognorm_mu, leaf.lognorm_sigma = 0.0, 1.0

def _get_pruning_sequence(node: NodeGPD) -> List[Tuple[float, NodeGPD]]:
    sequence = []
    def cost_and_leaves(n: NodeGPD) -> Tuple[float, int]:
        if n.is_leaf: return n.nll, 1
        left_nll, left_leaves = cost_and_leaves(n.left)
        right_nll, right_leaves = cost_and_leaves(n.right)
        subtree_nll, subtree_leaves = left_nll + right_nll, left_leaves + right_leaves
        if subtree_leaves > 1 and np.isfinite(n.nll) and np.isfinite(subtree_nll):
            alpha = (n.nll - subtree_nll) / (subtree_leaves - 1)
            if alpha >= 0: sequence.append((alpha, n))
        return subtree_nll, subtree_leaves
    if not node.is_leaf:
        cost_and_leaves(node)
        sequence.sort(key=lambda item: item[0])
    return sequence

def get_subtree_nll(tree: NodeGPD, X: np.ndarray, y: np.ndarray) -> float:
    if len(X) == 0: return 0.0
    leaf_assignments = [find_leaf(tree, x_i) for x_i in X]
    unique_leaves = list(set(leaf_assignments))
    total_nll = 0.0
    for leaf in unique_leaves:
        if leaf.gpd_params is None: return C.LARGE_FLOAT
        mask = [id(la) == id(leaf) for la in leaf_assignments]
        total_nll += _neg_loglik_gpd(leaf.gpd_params, y[mask])
    return total_nll if np.isfinite(total_nll) else C.LARGE_FLOAT

def prune_gpd_with_cv(root: NodeGPD, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> Tuple[NodeGPD, float]:
    if len(y) < n_folds * C.GPD_MIN_LEAF_DEFAULT:
        logging.warning("데이터가 부족하여 가지치기를 건너뜁니다.")
        return root, 0.0
    try:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=C.SEED)
        folds = list(kf.split(X, (y > np.median(y)).astype(int)))
    except ValueError:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=C.SEED)
        folds = list(kf.split(X))
    pruning_seq = _get_pruning_sequence(root)
    alphas = sorted(list(set([alpha for alpha, _ in pruning_seq if alpha < C.LARGE_FLOAT])))
    candidate_alphas = [0.0] + alphas + ([alphas[-1] * 1.1] if alphas else [])
    cv_scores_avg_nll = {alpha: [] for alpha in candidate_alphas}
    for _, test_indices in folds:
        X_test, y_test = X[test_indices], y[test_indices]
        if len(y_test) == 0: continue
        for alpha in candidate_alphas:
            pruned_tree = copy.deepcopy(root)
            while True:
                seq = _get_pruning_sequence(pruned_tree)
                if not seq or seq[0][0] >= alpha: break
                seq[0][1].is_leaf = True; seq[0][1].left = None; seq[0][1].right = None
            test_nll = get_subtree_nll(pruned_tree, X_test, y_test)
            if np.isfinite(test_nll): cv_scores_avg_nll[alpha].append(test_nll / len(y_test))
    mean_scores = {a: np.mean(s) for a, s in cv_scores_avg_nll.items() if s}
    if not mean_scores:
        logging.error("CV 가지치기에 실패했습니다.")
        return root, 0.0
    best_alpha = min(mean_scores, key=mean_scores.get)
    final_tree = copy.deepcopy(root)
    while True:
        seq = _get_pruning_sequence(final_tree)
        if not seq or seq[0][0] >= best_alpha: break
        seq[0][1].is_leaf = True; seq[0][1].left = None; seq[0][1].right = None
    return final_tree, best_alpha

def validate_tree_structure(tree: NodeBase, min_samples_leaf: int = 1) -> bool:
    valid, nodes = True, [tree]
    while nodes:
        node = nodes.pop(0)
        if node is None or not hasattr(node, 'is_leaf'):
            logging.error("유효하지 않은 노드 발견."); return False
        if node.is_leaf:
            if hasattr(node, 'n_samples') and 0 < node.n_samples < min_samples_leaf:
                logging.warning(f"리프 노드의 샘플 수가 최소치({min_samples_leaf})보다 적습니다: {node.n_samples}")
        else:
            if node.left is None or node.right is None or node.split_var is None:
                logging.error("내부 노드에 자식 또는 분기 정보가 없습니다."); return False
            nodes.extend([node.left, node.right])
    return valid

def print_tree_structure(node: NodeBase, feat_names: List[str], indent: str = ""):
    if node is None: return
    if node.is_leaf:
        n_str = f"N={node.n_samples}"
        if isinstance(node, NodeL1): print(f"{indent}Leaf: median={node.median_val:.3f}, {n_str}")
        elif isinstance(node, NodeL2): print(f"{indent}Leaf: mean={node.mean_val:.3f}, {n_str}")
        elif isinstance(node, NodeGPD):
            params_str = "params=None"
            if node.gpd_params:
                params_str = f"σ={node.gpd_params[0]:.3f}, γ={node.gpd_params[1]:.3f}"
            print(f"{indent}Leaf: GPD({params_str}), NLL={node.nll:.2f}, {n_str}")
        else: print(f"{indent}Leaf: {n_str}")
        return
    feat = feat_names[node.split_var]
    gain_str = ""
    if hasattr(node, 'gain') and node.gain > 0: gain_str = f"(Gain={node.gain:.3f})"
    elif hasattr(node, 'split_gain'): gain_str = f"(PenalizedGain={node.split_gain:.3f})"
    print(f"{indent}[{feat} <= {node.split_thr:.4g}] {gain_str} N={node.n_samples}")
    print(f"{indent}├─ True:", end=""); print_tree_structure(node.left, feat_names, indent + "│  ")
    print(f"{indent}└─ False:", end=""); print_tree_structure(node.right, feat_names, indent + "│  ")

# =====================================================================
# 4. 메인 실행 블록 (사용자 제공)
# =====================================================================

if __name__ == "__main__":
    # 이 블록은 사용자가 제공한 코드를 기반으로 하며,
    # 위에서 수정한 모듈 함수들을 사용하여 실행됩니다.
    
    # --- 설정 ---
    TARGET_VARIABLE = "Exceedance"
    DATE_COLUMN = "Date"
    START_YEAR = 1985

    selected_lags_dict = {
        'Three_Component_Index': [1], 'News_Based_Policy_Uncert_Index': [1],
        "1. Economic Policy Uncertainty": [1], '2. Monetary policy': [2],
        "Fiscal Policy (Taxes OR Spending)": [1], "3. Taxes": [1],
        "5. Health care": [1], "6. National security": [1],
        "7. Entitlement programs": [1], "8. Regulation": [1],
        "Financial Regulation": [1], "9. Trade policy": [2],
        "10. Sovereign debt, currency crises": [3],
        "House Price Index_diff": [2], "CPI_diff": [2], "Close_diff": [2]
    }

    categorical_cols_list = [
        'Category_Business Disruption and System Failures', 'Category_Clients, Products & Business Practices',
        'Category_Damage to Physical Assets', 'Category_Employment Practices and Workplace Safety',
        'Category_Execution, Delivery & Process Management', 'Category_External Fraud', 'Category_Internal Fraud',
        'Category_Commercial Banking', 'Category_Corporate Finance', 'Category_Health Insurance',
        'Category_Life Insurance and Benefit Plans', 'Category_Merchant Banking', 'Category_Municipal/Government Finance',
        'Category_Private Banking', 'Category_Property and Casualty Insurance',
        'Category_Reinsurance', 'Category_Retail Banking'
    ]

    # --- 데이터 로딩 (데모용) ---
    try:
        if 'df_merged' not in locals() or not isinstance(df_merged, pd.DataFrame):
            raise NameError("'df_merged' not found")
    except (NameError, FileNotFoundError):
        logging.warning("'df_merged' 데이터프레임을 찾을 수 없습니다. 데모용 합성 데이터를 생성합니다.")
        n_rows = 1000
        dates = pd.to_datetime(pd.date_range(start=f'{START_YEAR-5}-01-01', periods=n_rows, freq='M'))
        data = {'Date': dates}
        all_needed_cols = list(selected_lags_dict.keys()) + categorical_cols_list + [TARGET_VARIABLE]
        # _diff 컬럼의 원본 컬럼 이름 추가
        base_diff_cols = {k.replace("_diff", "") for k in selected_lags_dict if k.endswith("_diff")}
        all_needed_cols.extend(list(base_diff_cols))
        
        for col in all_needed_cols:
            if col not in data: # 중복 생성 방지
                if col == TARGET_VARIABLE:
                    data[col] = np.random.pareto(a=1.5, size=n_rows) * 1000 + 1
                elif col.startswith("Category_"):
                    data[col] = np.random.randint(0, 2, size=n_rows)
                else:
                    data[col] = np.random.randn(n_rows).cumsum() + 100
        df_merged = pd.DataFrame(data)

    # --- 피처 엔지니어링 ---
    try:
        df_model_ready, feature_names_list = create_lagged_features(
            df=df_merged,
            target_col=TARGET_VARIABLE,
            lag_config=selected_lags_dict,
            date_col=DATE_COLUMN,
            start_year=START_YEAR,
            additional_features=categorical_cols_list
        )
    except Exception as e:
        logging.error(f"시차 변수 생성 중 오류 발생: {e}")
        raise

    # --- 모델링 ---
    if df_model_ready.empty:
        logging.error("처리 후 데이터프레임이 비어 있습니다. 입력 데이터와 파라미터를 확인하세요.")
    else:
        X_all_data = df_model_ready[feature_names_list].values
        y_all_data = df_model_ready[TARGET_VARIABLE].values
        
        u_threshold = 191
        mask_bulk = y_all_data <= u_threshold
        mask_tail = ~mask_bulk

        X_bulk, y_bulk = X_all_data[mask_bulk], y_all_data[mask_bulk]
        X_tail, y_tail_excess = X_all_data[mask_tail], y_all_data[mask_tail] - u_threshold

        logging.info(f"데이터 분할: Body = {len(y_bulk)} 샘플, Tail (초과분) = {len(y_tail_excess)} 샘플.")

        # L1 CART (Body)
        logging.info("Fitting L1 CART (Median) for Bulk Data...")
        l1_bulk_tree = grow_tree_l1(X=X_bulk, y=y_bulk, feature_names=feature_names_list, min_leaf=20, max_depth=3)
        assign_lognorm_params(l1_bulk_tree, X_bulk, y_bulk, trunc_left=u_threshold)
        print("\n===== L1 CART (Body, Median-based) =====")
        print_tree_structure(l1_bulk_tree, feature_names_list)
        validate_tree_structure(l1_bulk_tree)

        # L2 CART (Body)
        logging.info("Fitting L2 CART (Mean) for Bulk Data...")
        l2_bulk_tree = grow_tree_l2(X=X_bulk, y=y_bulk, feature_names=feature_names_list, min_leaf=30, max_depth=4)
        assign_lognorm_params(l2_bulk_tree, X_bulk, y_bulk, trunc_left=u_threshold)
        print("\n===== L2 CART (Body, Mean-based) =====")
        print_tree_structure(l2_bulk_tree, feature_names_list)
        validate_tree_structure(l2_bulk_tree)

        # GPD CART (Tail)
        if len(y_tail_excess) >= C.GPD_MIN_LEAF_DEFAULT:
            logging.info("Fitting GPD CART for Tail Data (Excesses)...")
            gpd_tail_tree_raw = grow_tree_gpd(X=X_tail, y=y_tail_excess, feature_names=feature_names_list, min_leaf=15, max_depth=4)
            print("\n===== Raw GPD CART (Tail, Excesses) =====")
            print_tree_structure(gpd_tail_tree_raw, feature_names_list)
            
            logging.info("Pruning GPD CART using Cross-Validation...")
            try:
                cv_folds = min(5, max(2, len(y_tail_excess) // 10))
                gpd_tail_tree_pruned, best_alpha_gpd = prune_gpd_with_cv(gpd_tail_tree_raw, X_tail, y_tail_excess, n_folds=cv_folds)
                logging.info(f"GPD CART Pruning complete. Best alpha = {best_alpha_gpd:.4g}")
                print("\n===== Pruned GPD CART (Tail, Excesses) =====")
                print_tree_structure(gpd_tail_tree_pruned, feature_names_list)
                validate_tree_structure(gpd_tail_tree_pruned, min_samples_leaf=10)
            except Exception as e:
                logging.error(f"GPD Tree Pruning failed: {e}. Using raw tree.")
        else:
            logging.warning("GPD CART 학습을 위한 Tail 데이터가 부족하여 건너뜁니다.")

        logging.info("Script finished.")
