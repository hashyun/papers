import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy import stats
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math

"""
Full L1/L2 (Lognormal body) + GPD-CART (tail) with robust cost–complexity pruning.
Key fixes vs. prior version:
 1) Pruning is now POST-ORDER (bottom-up) and uses strict comparison.
 2) Cost-as-leaf includes alpha * 1.0 (was missing before).
 3) CV alpha grid includes 0.0 (no-prune option). Optional 1-SE rule + root-guard.

Run as a script to see a demo with synthetic data if df_merged is absent.
"""

# ---------------------------------------------------------------------#
# Logging Setup
# ---------------------------------------------------------------------#
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# ---------------------------------------------------------------------#
# Constants
# ---------------------------------------------------------------------#
FLT_INFO = np.finfo(float)

SEED        = 42
rng         = np.random.default_rng(SEED)
MIN_FLOAT   = FLT_INFO.eps
LARGE_FLOAT = 1e30

# ---------------------------------------------------------------------#
# 0. Utility Functions
# ---------------------------------------------------------------------#
def mse_loss(y: np.ndarray, mu: float) -> float:
    """Calculate Sum of Squared Errors."""
    if len(y) == 0:
        return 0.0
    return float(np.sum((y - mu) ** 2))


def mae_loss(y: np.ndarray, md: float) -> float:
    """Calculate Sum of Absolute Errors."""
    if len(y) == 0:
        return 0.0
    return float(np.sum(np.abs(y - md)))

# ---------------------------------------------------------------------#
# Base Node Class and Specific Nodes
# ---------------------------------------------------------------------#
class NodeBase:
    """Base class for all nodes in the trees."""
    __slots__ = ("is_leaf", "split_var", "split_thr", "gain", "left", "right", "depth", "n_samples", "split_cats")
    def __init__(self):
        self.is_leaf: bool = True
        self.split_var: Optional[int] = None
        self.split_thr: Optional[float] = None
        self.gain: float = 0.0
        self.left: Optional['NodeBase'] = None
        self.right: Optional['NodeBase'] = None
        self.depth: int = 0
        self.n_samples: int = 0
        self.split_cats = None  # For categorical splits: set of categories for left branch


class NodeL1(NodeBase):
    """Node for L1 CART (median-based)."""
    __slots__ = ("median_val", "lognorm_mu", "lognorm_sigma")

    def __init__(self):
        super().__init__()
        self.median_val: Optional[float] = None
        self.lognorm_mu: Optional[float] = None
        self.lognorm_sigma: Optional[float] = None


class NodeL2(NodeBase):
    """Node for L2 CART (mean-based)."""
    __slots__ = ("mean_val", "lognorm_mu", "lognorm_sigma")

    def __init__(self):
        super().__init__()
        self.mean_val: Optional[float] = None
        self.lognorm_mu: Optional[float] = None
        self.lognorm_sigma: Optional[float] = None


class NodeGPD(NodeBase):
    """Node for GPD CART."""
    __slots__ = ("gpd_params", "nll", "split_gain")

    def __init__(self):
        super().__init__()
        self.split_gain: float = 0.0
        self.gpd_params: Optional[Tuple[float, float]] = None  # (sigma, gamma)
        self.nll: Optional[float] = None  # Negative Log-Likelihood at this node


# ---------------------------------------------------------------------#
# 1. Distribution Fitting Functions (Lognormal, GPD)
# ---------------------------------------------------------------------#

def _nll_lognormal(params: Tuple[float, float],
                   y: np.ndarray,
                   *, trunc_right: float = 0.0) -> float:
    """Negative Log-Likelihood for (optionally right-truncated) LogNormal."""
    mu, sigma = params
    if sigma <= MIN_FLOAT or not np.isfinite(mu) or not np.isfinite(sigma):
        return LARGE_FLOAT

    # Right-truncation: keep only y <= u for fitting if trunc_right > 0
    if trunc_right and trunc_right > 0:
        y = y[y <= trunc_right]

    y_pos = y[y > 0]
    if y_pos.size == 0:
        return LARGE_FLOAT

    log_y = np.log(y_pos)
    z = (log_y - mu) / sigma

    # Base NLL (non-truncated)
    base = float(np.sum(np.log(sigma) + 0.5*np.log(2*np.pi) + 0.5*z**2 + log_y))

    # Add truncation correction: + n * log F(u)
    if trunc_right and trunc_right > 0:
        zc = (np.log(trunc_right) - mu) / sigma
        F = stats.norm.cdf(zc)
        F = np.clip(F, MIN_FLOAT, 1.0)
        base += y_pos.size * float(np.log(F))

    return base if np.isfinite(base) else LARGE_FLOAT
def gpd_nll(y: np.ndarray, sigma: float, gamma: float) -> float:
    """
    Vectorized GPD negative log-likelihood for exceedances y>0.
    Param constraints: sigma>0; 1 + gamma*y/sigma > 0 for all y.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return 0.0
    if not np.isfinite(sigma) or sigma <= 0.0 or not np.isfinite(gamma):
        return np.inf

    z = 1.0 + gamma * (y / sigma)
    if np.any(z <= 0.0):
        return np.inf

    n = y.size
    if abs(gamma) < 1e-12:
        # Limit gamma -> 0: Exponential
        return n * np.log(sigma) + np.sum(y) / sigma

    return n * np.log(sigma) + (1.0 + 1.0 / gamma) * np.sum(np.log(z))


# ---------------------------------------------------------------------
# 2) Numerical Hessian (central differences) for 2D params (sigma, gamma)
# ---------------------------------------------------------------------
def _numerical_hessian_2d(
    f: Callable[[np.ndarray], float],
    theta: np.ndarray,
    rel_eps: float = 1e-4
) -> np.ndarray:
    """
    Central-difference Hessian for 2D parameter vector.
    rel_eps: relative step size per parameter.
    """
    theta = np.asarray(theta, dtype=float)
    assert theta.size == 2
    H = np.zeros((2, 2), dtype=float)

    # Step sizes per parameter (respect sign/magnitude)
    eps = np.where(theta != 0.0, np.abs(theta) * rel_eps, rel_eps)

    # Points
    t = theta
    e1 = np.array([eps[0], 0.0])
    e2 = np.array([0.0, eps[1]])

    f00 = f(t)
    fpp = f(t + e1 + e2)
    fpm = f(t + e1 - e2)
    fmp = f(t - e1 + e2)
    fmm = f(t - e1 - e2)
    fpp1 = f(t + e1)
    fmm1 = f(t - e1)
    fpp2 = f(t + e2)
    fmm2 = f(t - e2)

    # Diagonals
    H[0, 0] = (fpp1 - 2.0 * f00 + fmm1) / (eps[0] ** 2)
    H[1, 1] = (fpp2 - 2.0 * f00 + fmm2) / (eps[1] ** 2)
    # Cross (symmetric)
    H12 = (fpp - fpm - fmp + fmm) / (4.0 * eps[0] * eps[1])
    H[0, 1] = H12
    H[1, 0] = H12
    return H


def _safe_inv_2x2(M: np.ndarray) -> Optional[np.ndarray]:
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if not np.isfinite(det) or abs(det) < 1e-12:
        return None
    inv = np.array([[ M[1, 1], -M[0, 1]],
                    [-M[1, 0],  M[0, 0]]], dtype=float) / det
    return inv




def fit_lognormal_mle(y: np.ndarray, *, trunc_right: float = 0.0) -> Tuple[float, float]:
    """MLE for (optionally right-truncated) LogNormal -> (μ̂, σ̂)."""
    if trunc_right and trunc_right > 0:
        y = y[y <= trunc_right]

    y_pos = y[y > 0]
    n_pos = y_pos.size
    if n_pos < 2:
        return (float(np.log(y_pos[0])), max(1e-2, MIN_FLOAT*10)) if n_pos == 1 else (0.0, 1.0)

    log_y = np.log(y_pos)
    init = [float(log_y.mean()), float(max(log_y.std(ddof=1), MIN_FLOAT*10))]
    bounds = [(None, None), (MIN_FLOAT*10, None)]

    obj = lambda p: _nll_lognormal(p, y_pos, trunc_right=trunc_right)

    try:
        res = minimize(obj, init, method="L-BFGS-B", bounds=bounds)
        if res.success and np.all(np.isfinite(res.x)):
            mu, sigma = res.x
            return float(mu), float(max(float(sigma), MIN_FLOAT*100))
    except Exception as e:
        logging.debug(f"LogNormal MLE failed: {e}")

    return tuple(init)  # type: ignore



def fit_gpd_mle(y: np.ndarray, *, gamma_bounds: Tuple[float, float] = (0.01, 3.0)) -> np.ndarray:
    """
    Stabilized GPD MLE with PWM init.
    논문 초점(heavy-tailed, γ>0)에 맞춰 γ 범위를 (0.01, 1.0)으로 제한.
    소표본(n<30) 폴백도 γ>0 유지.
    """
    n = len(y)
    if n < 30:
        fallback_sigma = max(float(np.median(y)), MIN_FLOAT * 10) if n > 0 else 1.0
        return np.array([fallback_sigma, 0.1], dtype=float)  # γ>0 폴백

    # PWM init (γ0 양수로 클리핑)
    y_sorted = np.sort(y)
    y_bar = float(np.mean(y_sorted))
    pwm1 = float(np.mean((1.0 - (np.arange(n) + 0.65) / n) * y_sorted))

    denom = (y_bar - 2.0 * pwm1 + MIN_FLOAT)
    gamma0_raw = 2.0 - y_bar / denom
    gamma0 = float(np.clip(gamma0_raw, gamma_bounds[0], gamma_bounds[1]))
    sigma0 = max((2.0 * y_bar * pwm1) / denom, MIN_FLOAT * 10)

    init_params = np.array([sigma0, gamma0], dtype=float)
    bounds = [(MIN_FLOAT*10, None), gamma_bounds]

    def objective(theta: np.ndarray, data: np.ndarray) -> float:
        return gpd_nll(data, float(theta[0]), float(theta[1]))

    try:
        res = minimize(
            objective, x0=init_params, args=(y,), method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-7},
        )
        if res.success and np.isfinite(res.fun):
            s_opt, g_opt = res.x
            return np.array([max(float(s_opt), MIN_FLOAT * 10), float(max(g_opt, gamma_bounds[0]))], dtype=float)
        else:
            return init_params
    except Exception:
        return init_params




def node_cost_gpd(y: np.ndarray) -> Tuple[float, np.ndarray]:
    """Calculate node cost (NLL) and parameters for GPD."""
    if len(y) == 0:
        # gamma=0.1로 기본값 (0도 가능하지만 0보다 약간 양수면 안정적)
        return 0.0, np.array([1.0, 0.1], dtype=float)

    try:
        theta = fit_gpd_mle(y)                     # theta = [sigma, gamma]
        sigma, gamma = float(theta[0]), float(theta[1])
        cost = gpd_nll(y, sigma, gamma)            # ← 인자 순서/형태 수정 (중요)
        if not np.isfinite(cost):
            return LARGE_FLOAT, theta
        return float(cost), np.array([sigma, gamma], dtype=float)
    except Exception as e:
        logging.warning(f"Error during node_cost_gpd: {e}. Returning large cost.")
        fallback_sigma = max(float(np.median(y)), MIN_FLOAT*10) if len(y) > 0 else 1.0
        return LARGE_FLOAT, np.array([fallback_sigma, 0.1], dtype=float)

# ---------------------------------------------------------------------#
# 2. CART Splitting Functions
# ---------------------------------------------------------------------#

def _best_split_l1(x: np.ndarray, y: np.ndarray, min_leaf: int = 1) -> Tuple[float, Optional[float]]:
    """Find the best split point for L1 CART (median-based)."""
    n = len(y)
    if n < 2 * min_leaf:
        return 0.0, None

    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]

    parent_loss = mae_loss(ys, float(np.median(ys)))
    best_gain, best_thr = 0.0, None

    for i in range(min_leaf, n - min_leaf):
        if xs[i] == xs[i - 1]:
            continue
        left_y, right_y = ys[:i], ys[i:]
        child_loss = mae_loss(left_y, float(np.median(left_y))) + mae_loss(right_y, float(np.median(right_y)))
        gain = parent_loss - child_loss
        if gain > best_gain:
            best_gain = float(gain)
            best_thr = 0.5 * (xs[i] + xs[i - 1])

    return (best_gain, best_thr) if best_gain > MIN_FLOAT else (0.0, None)


def _best_split_l2(x: np.ndarray, y: np.ndarray, min_leaf: int = 1) -> Tuple[float, Optional[float]]:
    """Find the best split point for L2 CART (mean-based) efficiently."""
    n = len(y)
    if n < 2 * min_leaf:
        return 0.0, None

    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]

    parent_loss = float(np.sum((ys - np.mean(ys)) ** 2))

    csum = np.cumsum(ys)
    csq = np.cumsum(ys ** 2)
    total_sum, total_sq = csum[-1], csq[-1]

    best_gain, best_thr = 0.0, None

    for i in range(min_leaf, n - min_leaf):
        if xs[i] == xs[i - 1]:
            continue

        cnt_l = i
        loss_l = csq[i-1] - (csum[i-1]**2) / cnt_l

        cnt_r = n - cnt_l
        loss_r = (total_sq - csq[i-1]) - ((total_sum - csum[i-1])**2) / cnt_r

        gain = parent_loss - float(loss_l + loss_r)
        if gain > best_gain:
            best_gain = float(gain)
            best_thr = 0.5 * (xs[i] + xs[i - 1])

    return (best_gain, best_thr) if best_gain > MIN_FLOAT else (0.0, None)


def _best_split_gpd(
    x: np.ndarray,
    y: np.ndarray,
    min_leaf: int = 100,
    gain_tol: float = 1e-8
) -> Tuple[float, Optional[object]]:
    """
    Returns: (best_gain, split_descriptor)
      - For numeric features: split_descriptor is a float threshold.
      - For categorical features: split_descriptor is a tuple ("cat", tuple_of_left_categories).
    Growth criterion is pure ΔNLL = parent_nll - (left_nll + right_nll).
    """
    n = len(y)
    if n < 2 * min_leaf:
        return 0.0, None

    x_arr = np.asarray(x)
    # Detect categorical: object/str or pandas categorical
    is_cat = (x_arr.dtype.kind in ("O", "U", "S"))

    # Helper: compute ΔNLL for a boolean mask (left=True)
    def _gain_from_mask(mask: np.ndarray) -> float:
        if mask.sum() < min_leaf or (n - mask.sum()) < min_leaf:
            return -np.inf
        left_y, right_y = y[mask], y[~mask]
        l_nll, _ = node_cost_gpd(left_y)
        r_nll, _ = node_cost_gpd(right_y)
        if not (np.isfinite(l_nll) and np.isfinite(r_nll)):
            return -np.inf
        return parent_nll - (l_nll + r_nll)

    parent_nll, _ = node_cost_gpd(y)
    if not np.isfinite(parent_nll):
        return 0.0, None

    if is_cat:
        # Categorical splitting
        x_flat = np.asarray(x_arr, dtype=object)
        # Handle NaNs/None by treating them as a separate category
        x_norm = np.array([("∅" if (xi is None or (isinstance(xi, float) and np.isnan(xi))) else xi) for xi in x_flat], dtype=object)
        # Unique categories and counts
        uniques, counts = np.unique(x_norm, return_counts=True)
        # If only 1 category → no split
        if uniques.size <= 1:
            return 0.0, None

        # Collapse ultra-rare categories into "OTHER" to stabilize if many tiny groups
        min_leaf_cat = max(1, min_leaf // 2)
        rare_mask = counts < min_leaf_cat
        if rare_mask.any():
            rare_set = set(uniques[rare_mask])
            x_collapsed = np.array([("OTHER" if (xi in rare_set) else xi) for xi in x_norm], dtype=object)
            uniques, counts = np.unique(x_collapsed, return_counts=True)
            x_use = x_collapsed
        else:
            x_use = x_norm

        k = uniques.size
        # Strategy A: order modalities by mean(y) and perform monotone (prefix) splits
        # Compute mean(y) per category for ordering
        cat_to_mean = {}
        for u in uniques:
            cat_to_mean[u] = float(np.mean(y[x_use == u])) if np.any(x_use == u) else -np.inf
        order = sorted(list(uniques), key=lambda u: cat_to_mean[u])
        best_gain, best_desc = 0.0, None

        # Evaluate prefix splits along the order
        left_set = set()
        for i in range(1, k):  # leave at least one category on the right
            left_set.add(order[i-1])
            mask = np.isin(x_use, list(left_set))
            gain = _gain_from_mask(mask)
            if gain > best_gain:
                best_gain = float(gain)
                best_desc = ("cat", tuple(sorted(left_set)))

        # Strategy B (rare): small k → brute-force subset search for better split
        MAX_BRUTE_K = 8
        if k <= MAX_BRUTE_K:
            # All non-empty, proper subsets
            # To avoid duplicate complements, we only iterate half the subsets by index convention
            import itertools as _it
            cats = list(uniques)
            for r in range(1, k // 2 + 1):
                for comb in _it.combinations(cats, r):
                    mask = np.isin(x_use, comb)
                    gain = _gain_from_mask(mask)
                    if gain > best_gain:
                        best_gain = float(gain)
                        best_desc = ("cat", tuple(sorted(comb)))

        if best_gain <= gain_tol:
            return 0.0, None
        return best_gain, best_desc

    # Numeric splitting (original behavior)
    idx = np.argsort(x_arr)
    xs, ys = x_arr[idx], y[idx]

    best_gain, best_thr = 0.0, None

    # Parent already computed above
    for i in range(min_leaf, n - min_leaf):
        if xs[i] == xs[i - 1]:
            continue
        left_y, right_y = ys[:i], ys[i:]
        l_nll, _ = node_cost_gpd(left_y)
        r_nll, _ = node_cost_gpd(right_y)
        if not (np.isfinite(l_nll) and np.isfinite(r_nll)):
            continue
        gain = parent_nll - (l_nll + r_nll)
        if gain > best_gain:
            best_gain = float(gain)
            best_thr = 0.5 * (float(xs[i]) + float(xs[i - 1]))

    if best_gain <= gain_tol:
        return 0.0, None
    return best_gain, best_thr


# ---------------------------------------------------------------------#
# 3. Tree Growing Functions
# ---------------------------------------------------------------------#

def grow_tree_l1(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                 min_leaf: int = 20, max_depth: Optional[int] = 3, depth: int = 0) -> NodeL1:
    """Grows an L1 CART (median-based) recursively."""
    n_samples, n_features = X.shape
    node = NodeL1()
    node.depth = depth
    node.n_samples = n_samples
    node.median_val = float(np.median(y)) if n_samples > 0 else 0.0

    stop_conditions = [
        n_samples == 0,
        n_samples < 2 * min_leaf,
        (max_depth is not None and depth >= max_depth)
    ]
    if any(stop_conditions):
        return node

    best_gain, best_var, best_thr = -1.0, None, None

    for j in range(n_features):
        gain, thr = _best_split_l1(X[:, j], y, min_leaf)
        if thr is not None and gain > best_gain:
            best_gain, best_var, best_thr = gain, j, thr

    if best_thr is None or best_gain <= MIN_FLOAT:
        return node

    mask_left = X[:, best_var] <= best_thr
    n_left, n_right = int(np.sum(mask_left)), n_samples - int(np.sum(mask_left))

    if n_left < min_leaf or n_right < min_leaf:
        return node

    node.is_leaf = False
    node.split_var, node.split_thr, node.gain = best_var, best_thr, float(best_gain)

    logging.debug(f"Depth {depth}: Splitting '{feature_names[best_var]}' <= {best_thr:.3g}, Gain={best_gain:.3f}, N={n_samples}->({n_left}, {n_right})")
    node.left = grow_tree_l1(X[mask_left], y[mask_left], feature_names, min_leaf, max_depth, depth + 1)
    node.right = grow_tree_l1(X[~mask_left], y[~mask_left], feature_names, min_leaf, max_depth, depth + 1)
    return node


def grow_tree_l2(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                 min_leaf: int = 30, max_depth: Optional[int] = 4, depth: int = 0) -> NodeL2:
    """Grows an L2 CART (mean-based) recursively."""
    n_samples, n_features = X.shape
    node = NodeL2()
    node.depth, node.n_samples = depth, n_samples
    node.mean_val = float(np.mean(y)) if n_samples > 0 else 0.0

    stop_conditions = [
        n_samples == 0,
        n_samples < 2 * min_leaf,
        (max_depth is not None and depth >= max_depth)
    ]
    if any(stop_conditions):
        return node

    best_gain, best_var, best_thr = -1.0, None, None

    for j in range(n_features):
        gain, thr = _best_split_l2(X[:, j], y, min_leaf)
        if thr is not None and gain > best_gain:
            best_gain, best_var, best_thr = gain, j, thr

    if best_thr is None or best_gain <= MIN_FLOAT:
        return node

    mask_left = X[:, best_var] <= best_thr
    n_left, n_right = int(np.sum(mask_left)), n_samples - int(np.sum(mask_left))

    if n_left < min_leaf or n_right < min_leaf:
        return node

    node.is_leaf = False
    node.split_var, node.split_thr, node.gain = best_var, best_thr, float(best_gain)

    logging.debug(f"Depth {depth}: Splitting '{feature_names[best_var]}' <= {best_thr:.3g}, Gain={best_gain:.3f}, N={n_samples}->({n_left}, {n_right})")
    node.left = grow_tree_l2(X[mask_left], y[mask_left], feature_names, min_leaf, max_depth, depth + 1)
    node.right = grow_tree_l2(X[~mask_left], y[~mask_left], feature_names, min_leaf, max_depth, depth + 1)
    return node


def grow_tree_gpd(X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]],
                  min_leaf: int = 20,            # 논문 권장 범위(실데이터 20, 시뮬 50)
                  max_depth: Optional[int] = None,
                  depth: int = 0) -> NodeGPD:
    """Grows a GPD CART recursively (paper-mode: pure ΔNLL splitting, no pre-pruning)."""
    n_samples, n_features = X.shape
    node = NodeGPD()
    node.depth, node.n_samples = depth, n_samples
    node.nll, node.gpd_params = node_cost_gpd(y)

    # 종료 조건: 표본/깊이 제약만 (pre-pruning 없음)
    if (n_samples == 0) or (n_samples < 2 * min_leaf) or (max_depth is not None and depth >= max_depth):
        logging.debug(f"[GROW] depth={depth}: Leaf (stop). N={n_samples}, NLL={node.nll:.2f}")
        return node

    if not np.isfinite(node.nll):
        logging.warning(f"[GROW] depth={depth}: Leaf (GPD fit failed). N={n_samples}")
        return node

    best_gain, best_var, best_thr = 0.0, None, None

    for j in range(n_features):
        # 논문식: 순수 ΔNLL → threshold=0을 보장하기 위해 delta_params=0
        gain, thr = _best_split_gpd(
            X[:, j], y,
            min_leaf=min_leaf,
         # ★ 핵심: pre-pruning 제거(ΔNLL만)
            gain_tol=1e-8
        )
        if thr is not None and gain > best_gain:
            best_gain, best_var, best_thr = float(gain), j, thr

    if best_thr is None or best_gain <= 0.0:
        logging.debug(f"[GROW] depth={depth}: Leaf (no ΔNLL-improving split). N={n_samples}, NLL={node.nll:.2f}")
        return node

    
    # Build split mask according to descriptor (numeric vs categorical)
    split_desc = best_thr
    if isinstance(split_desc, tuple) and len(split_desc) >= 2 and split_desc[0] == "cat":
        left_cats = set(split_desc[1])
        mask_left = np.isin(X[:, best_var], list(left_cats))
    else:
        # numeric threshold
        mask_left = X[:, best_var] <= float(split_desc)

    n_left = int(np.sum(mask_left))
    n_right = n_samples - n_left
    if (n_left < min_leaf) or (n_right < min_leaf):
        logging.debug(f"[GROW] depth={depth}: Leaf (min_leaf violated). N={n_samples}, NLL={node.nll:.2f}")
        return node

    node.is_leaf = False
    node.split_var = best_var
    if isinstance(split_desc, tuple) and split_desc[0] == "cat":
        node.split_thr = None
        node.split_cats = set(split_desc[1])
        split_info_str = f"{feature_names[best_var] if feature_names else f'Feature_{best_var}'} ∈ {sorted(list(node.split_cats))}"
    else:
        node.split_thr = float(split_desc)
        node.split_cats = None
        split_info_str = f"{feature_names[best_var] if feature_names else f'Feature_{best_var}'} ≤ {node.split_thr:.6g}"
    node.split_gain = float(best_gain)

    logging.debug(
        f"[GROW] depth={depth}: split {split_info_str}, ΔNLL={best_gain:.6f}, N={n_samples} -> ({n_left}, {n_right})"
    )

    node.left  = grow_tree_gpd(X[mask_left],  y[mask_left],  feature_names, min_leaf, max_depth, depth + 1)
    node.right = grow_tree_gpd(X[~mask_left], y[~mask_left], feature_names, min_leaf, max_depth, depth + 1)
    return node
    


# ---------------------------------------------------------------------#
# 4. Leaf Finding and Parameter Assignment
# ---------------------------------------------------------------------#


def find_leaf(node, x):
    """Robust leaf finder: supports numeric thresholds and categorical splits."""
    current = node
    while (current is not None) and (not current.is_leaf):
        if (current.split_var is None) or (current.left is None) or (current.right is None):
            break
        try:
            if getattr(current, "split_cats", None) is not None:
                go_left = (x[current.split_var] in current.split_cats)
            else:
                thr = current.split_thr
                if thr is None:
                    break
                go_left = (x[current.split_var] <= thr)
        except Exception:
            break
        next_node = current.left if go_left else current.right
        if next_node is None:
            break
        current = next_node
    return current



def _get_all_leaves(node: Union[NodeL1, NodeL2, NodeGPD]) -> List[Union[NodeL1, NodeL2, NodeGPD]]:
    """Recursively collect all leaf nodes."""
    leaves: List[Union[NodeL1, NodeL2, NodeGPD]] = []
    if node.is_leaf:
        leaves.append(node)
    else:
        if node.left: leaves.extend(_get_all_leaves(node.left))
        if node.right: leaves.extend(_get_all_leaves(node.right))
    return leaves


def _num_leaves(node: NodeBase) -> int:
    return len(_get_all_leaves(node))


def assign_lognorm_params(root: Union['NodeL1', 'NodeL2'],
                          X: np.ndarray, y: np.ndarray,
                          *, trunc_right: Optional[float] = None) -> None:
    """
    각 leaf에 로그노말(μ,σ) 적합. trunc_right=u를 주면 y<=u 바디만 사용.
    (PoT: 바디는 u 이하, 테일은 u 초과를 GPD로)
    """
    if root is None:
        return
    leaves = _get_all_leaves(root)
    if not leaves:
        return

    leaf_map = {id(l): i for i, l in enumerate(leaves)}
    leaf_indices = np.array([leaf_map[id(find_leaf(root, x))] for x in X])

    for i, lf in enumerate(leaves):
        y_leaf_full = y[leaf_indices == i]
        y_leaf = y_leaf_full if (trunc_right is None or trunc_right <= 0) else y_leaf_full[y_leaf_full <= trunc_right]
        lf.n_samples = y_leaf_full.size
        if y_leaf.size > 1:
            mu, sigma = fit_lognormal_mle(y_leaf, trunc_right=float(trunc_right) if trunc_right else 0.0)
            lf.lognorm_mu, lf.lognorm_sigma = float(mu), float(sigma)
        else:
            lf.lognorm_mu, lf.lognorm_sigma = 0.0, 1.0
        if isinstance(lf, NodeL1):
            lf.median_val = float(np.median(y_leaf_full)) if y_leaf_full.size > 0 else 0.0
        if isinstance(lf, NodeL2):
            lf.mean_val   = float(np.mean(y_leaf_full)) if y_leaf_full.size > 0 else 0.0


# ---------------------------------------------------------------------#
# 5. GPD Tree Pruning (Cost-Complexity Pruning with CV)
# ---------------------------------------------------------------------#

def get_subtree_nll(tree: Any, X: np.ndarray, y_excess: np.ndarray) -> float:
    """
    (패치) 리프별로 y를 묶어 gpd_nll을 배치로 계산하여 총 NLL을 반환.
    X, y_excess 는 동일 샘플 순서여야 하고 y_excess 는 (y - u) 초과치.
    """
    if X is None or y_excess is None or len(X) == 0:
        return 0.0

    X = np.asarray(X)
    y = np.asarray(y_excess, dtype=float)

    # 모듈 내 find_leaf / gpd_nll / LARGE_FLOAT 사용
    unique_ids, groups_idx, id2leaf, _ = _group_by_leaf_ids(tree, X, find_leaf=find_leaf)

    total = 0.0
    for grp_idx, oid in zip(groups_idx, unique_ids):
        lf = id2leaf.get(oid, None)
        if (lf is None) or (getattr(lf, "gpd_params", None) is None):
            return LARGE_FLOAT
        sigma, gamma = float(lf.gpd_params[0]), float(lf.gpd_params[1])
        y_grp = y[grp_idx]
        nll = gpd_nll(y_grp, sigma, gamma)
        total += nll if np.isfinite(nll) else LARGE_FLOAT

    return float(total)




def _compute_max_depth(node: NodeBase) -> int:
    """Return the maximum depth among leaves in this tree (root.depth is 0)."""
    if node is None:
        return 0
    if node.is_leaf:
        return int(getattr(node, "depth", 0))
    return max(_compute_max_depth(node.left), _compute_max_depth(node.right))


def _repair_tree_inplace(node: NodeBase) -> None:
    """Fix malformed internal nodes by converting them to leaves."""
    if node is None:
        return
    if node.is_leaf:
        return
    broken = (
        (node.left is None) or (node.right is None) or
        (node.split_var is None) or (node.split_thr is None)
    )
    if broken:
        node.is_leaf = True
        node.left = None
        node.right = None
        node.split_var = None
        node.split_thr = None
        if hasattr(node, "split_gain"):
            node.split_gain = 0.0
        return
    _repair_tree_inplace(node.left)
    _repair_tree_inplace(node.right)


def _get_pruning_sequence(node: NodeGPD) -> List[Tuple[float, NodeGPD]]:
    """Generates (alpha, node) tuples for cost-complexity pruning."""
    sequence: List[Tuple[float, NodeGPD]] = []

    def get_leaves_and_cost(n: Optional[NodeGPD]):
        if n is None:
            return 0, 0.0
        if n.is_leaf:
            return 1, n.nll if (n.nll is not None and np.isfinite(n.nll)) else LARGE_FLOAT

        left_leaves, left_nll = get_leaves_and_cost(n.left)
        right_leaves, right_nll = get_leaves_and_cost(n.right)

        total_leaves = left_leaves + right_leaves
        total_leaf_nll = left_nll + right_nll

        if total_leaves > 1 and np.isfinite(total_leaf_nll) and np.isfinite(n.nll):
            alpha = (n.nll - total_leaf_nll) / (total_leaves - 1)
            if alpha >= -MIN_FLOAT:
                sequence.append((float(alpha), n))

        return total_leaves, total_leaf_nll

    _repair_tree_inplace(node)
    if (node is not None) and (not node.is_leaf):
        get_leaves_and_cost(node)
        sequence.sort(key=lambda item: item[0])

    return sequence


def prune_gpd_single_alpha(tree_root: NodeGPD, alpha: float) -> NodeGPD:
    """Bottom-up (post-order) cost–complexity pruning at a fixed alpha."""
    pruned = copy.deepcopy(tree_root)
    TOL = 1e-8

    def _postorder(node: Optional[NodeGPD]) -> Optional[NodeGPD]:
        if node is None or node.is_leaf:
            return node

        # prune children first
        node.left = _postorder(node.left)
        node.right = _postorder(node.right)

        # compare subtree vs leaf
        leaves = _get_all_leaves(node)
        cost_subtree = sum(
            lf.nll for lf in leaves
            if (lf is not None) and (lf.nll is not None) and np.isfinite(lf.nll)
        )
        num_leaves = len(leaves)

        cost_as_leaf = (node.nll if (node.nll is not None) else LARGE_FLOAT) + alpha * 1.0
        cost_as_subtree = cost_subtree + alpha * num_leaves

        # strict improvement only; ties keep structure
        if cost_as_leaf + TOL < cost_as_subtree:
            node.is_leaf = True
            node.left = node.right = None
            node.split_var = node.split_thr = None
            if hasattr(node, "split_gain"):
                node.split_gain = 0.0
        return node

    return _postorder(pruned)  # type: ignore


def _alpha_root_threshold(root: NodeGPD) -> float:
    """Alpha at which the root would collapse to a single leaf."""
    leaves = _get_all_leaves(root)
    if len(leaves) <= 1 or root.nll is None or not np.isfinite(root.nll):
        return math.inf
    leaf_sum = sum(lf.nll for lf in leaves if lf.nll is not None and np.isfinite(lf.nll))
    denom = max(len(leaves) - 1, 1)
    return float((root.nll - leaf_sum) / denom)



def _alpha_grid_from_rate(k_excess: int,
                          *,
                          c_low: float = 0.5,
                          c_high: float = 5.0,
                          n_grid: int = 25) -> np.ndarray:
    """
    프루닝 페널티의 이론적 스케일 ~ sqrt(log k)/sqrt(k) 주변에서 α 그리드 생성.
    k_excess: 초과치 개수 k (tail 표본 수)
    """
    if k_excess <= 0:
        return np.array([0.0], dtype=float)
    base = np.sqrt(np.log(max(k_excess, 2)) / max(k_excess, 1))
    lo = max(1e-8, c_low * base)
    hi = max(lo * 1.5, c_high * base)
    return np.unique(np.concatenate([[0.0], np.linspace(lo, hi, n_grid)])).astype(float)
    


def _suggest_min_leaf(k_excess: int) -> int:
    """
    초과치 수 k에 연동된 리프 최소 표본수 추천값.
    너무 작은 리프(추정분산 ↑)와 과도한 분할을 억제.
    """
    return max(30, int(np.ceil(2.0 * np.log(max(k_excess, 2)))))
   
def warn_leaf_moments(tree: 'NodeGPD', *,
                      mean_guard: float = 1.0,
                      near_tol: float = 0.95) -> None:
    """
    각 리프의 GPD γ를 확인하여 평균 존재(γ<1) 여부 경고.
    γ ≥ 1 → 평균 무한 (보험가능성 리스크), γ≈1 근접 경고.
    """
    for lf in _get_all_leaves(tree):
        params = getattr(lf, "gpd_params", None)
        if params is None or len(params) < 2:
            continue
        sigma, gamma = float(params[0]), float(params[1])
        d = getattr(lf, "depth", "?")
        if gamma >= mean_guard:
            logging.warning(f"[Leaf depth={d}] γ={gamma:.3f} ≥ 1 → mean is infinite; insurability risk.")
        elif gamma >= near_tol:
            logging.info(f"[Leaf depth={d}] γ={gamma:.3f} near 1; mean near-divergent (주의).")


    

def prune_gpd_with_cv(
    root: 'NodeGPD', X: np.ndarray, y: np.ndarray, *,
    n_folds: Optional[int] = None,
    random_state: int = 42,
    cv_max_depth: Optional[int] = None,
    cv_min_leaf_ratio: float = 0.05,
    cv_min_leaf_floor: int = 30,
    use_one_se: bool = True,
    use_mean_nll: bool = True,
    guard_root: bool = True,
    copy_tree_each_alpha: bool = True,  # in-place 프루닝 보호
) -> Tuple['NodeGPD', float]:
    """
    논문식 CV:
      점수(α) = 검증 평균 NLL + α * K(리프수)
      α 후보: {0} ∪ pruning path α들 ∪ logspace 보강 ∪ rate grid  (한 번만 unique/sort)
    """
    from sklearn.model_selection import KFold
    from copy import deepcopy

    n = len(y)
    if n_folds is None:
        n_folds = max(3, min(5, n // 100))
    if n < max(50, n_folds * 10):
        logging.warning("Tail too small for CV pruning; returning unpruned tree.")
        return root, 0.0

    _repair_tree_inplace(root)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # --- α grid: 한 번만 구성/정리 ---
    seq = _get_pruning_sequence(root)
    alphas_root = sorted({float(a) for a, _ in seq if np.isfinite(a) and a < LARGE_FLOAT})
    rate_grid = _alpha_grid_from_rate(len(y))
    alpha_grid_parts = [np.array([0.0], dtype=float), np.asarray(rate_grid, dtype=float)]
    if alphas_root:
        alpha_min, alpha_max = min(alphas_root), max(alphas_root)
        alpha_grid_parts.append(np.array(alphas_root, dtype=float))
        alpha_grid_parts.append(
            np.logspace(np.log10(max(alpha_min, 1e-8)),
                        np.log10(alpha_max * 5.0 + 1e-12), 30).astype(float)
        )
    alpha_grid = np.unique(np.concatenate(alpha_grid_parts)).astype(float)
    alpha_grid = alpha_grid[np.isfinite(alpha_grid)]
    alpha_grid = alpha_grid[alpha_grid >= 0]
    alpha_grid.sort()
    if alpha_grid.size == 0:
        alpha_grid = np.array([0.0], dtype=float)

    scores: Dict[float, List[float]] = {float(a): [] for a in alpha_grid}

    # --- CV 루프 ---
    for train_idx, test_idx in kf.split(X):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        # fold별 min_leaf 설정
        min_leaf_auto = _suggest_min_leaf(len(y_tr))
        min_leaf_ratio_based = int(cv_min_leaf_ratio * len(y_tr))
        min_leaf = max(min_leaf_auto, min_leaf_ratio_based, cv_min_leaf_floor)

        # 최대 트리 성장 (pre-pruning 없음)
        tree_fold = grow_tree_gpd(X_tr, y_tr, feature_names=None,
                                  min_leaf=min_leaf, max_depth=cv_max_depth)
        _repair_tree_inplace(tree_fold)

        for a in alpha_grid:
            if copy_tree_each_alpha:
                pruned = prune_gpd_single_alpha(deepcopy(tree_fold), float(a))
            else:
                pruned = prune_gpd_single_alpha(tree_fold, float(a))
            _repair_tree_inplace(pruned)

            # 점수 = (평균 NLL 또는 총 NLL) + α*K
            total_nll = get_subtree_nll(pruned, X_te, y_te)
            mean_nll = total_nll / max(len(y_te), 1)
            K = _num_leaves(pruned)
            score_core = (mean_nll if use_mean_nll else total_nll)
            score = score_core + float(a) * K
            if np.isfinite(score):
                scores[float(a)].append(float(score))

    means = {a: float(np.mean(v)) for a, v in scores.items() if len(v) == n_folds}
    if not means:
        logging.error("CV pruning failed (insufficient filled folds). Returning unpruned tree.")
        return root, 0.0

    ses = {a: float(np.std(scores[a], ddof=1) / np.sqrt(n_folds)) for a in means}
    alpha_min_mean = min(means, key=means.get)
    if use_one_se:
        threshold = means[alpha_min_mean] + ses.get(alpha_min_mean, 0.0)
        candidates = [a for a, m in means.items() if m <= threshold]
        best_alpha = max(candidates) if candidates else alpha_min_mean
    else:
        best_alpha = alpha_min_mean

    if guard_root:
        if ' _alpha_root_threshold' in globals():
            ar = _alpha_root_threshold(root)
        else:
            ar = np.inf
        if np.isfinite(ar) and best_alpha >= ar:
            smaller = [a for a in alpha_grid if a < ar]
            if smaller:
                best_alpha = max(smaller)

    final_pruned = prune_gpd_single_alpha(root, float(best_alpha))
    _repair_tree_inplace(final_pruned)

    logging.info(
        f"CV(paper): best_alpha={best_alpha:.4g}, "
        f"leaves_raw={_num_leaves(root)}, leaves_pruned={_num_leaves(final_pruned)}, "
        f"mean_score_min={means[best_alpha]:.6g}"
    )
    return final_pruned, float(best_alpha)





# ---------------------------------------------------------------------#
# 6. Tree Visualization and Evaluation
# ---------------------------------------------------------------------#

def print_tree_structure(node: Union[NodeL1, NodeL2, NodeGPD],
                         feat_names: Optional[List[str]] = None, indent: str = ""):
    """Prints the structure of any supported tree type."""
    if not hasattr(node, 'is_leaf'):
        print(f"{indent}Error: Invalid node object.")
        return

    n_samples_str = f"N={getattr(node, 'n_samples', '?')}"

    if node.is_leaf:
        prefix = f"{indent}Leaf: {n_samples_str}"
        if isinstance(node, NodeL1):
            median_str = f"{node.median_val:.2f}" if node.median_val is not None else "N/A"
            mu_str = f"{node.lognorm_mu:.2f}" if node.lognorm_mu is not None else "N/A"
            sigma_str = f"{node.lognorm_sigma:.2f}" if node.lognorm_sigma is not None else "N/A"
            print(f"{prefix}, median={median_str}, LN(mu={mu_str}, sigma={sigma_str})")
        elif isinstance(node, NodeL2):
            mean_str = f"{node.mean_val:.2f}" if node.mean_val is not None else "N/A"
            mu_str = f"{node.lognorm_mu:.2f}" if node.lognorm_mu is not None else "N/A"
            sigma_str = f"{node.lognorm_sigma:.2f}" if node.lognorm_sigma is not None else "N/A"
            print(f"{prefix}, mean={mean_str}, LN(mu={mu_str}, sigma={sigma_str})")
        elif isinstance(node, NodeGPD):
            nll_str = f"{node.nll:.1f}" if node.nll is not None and np.isfinite(node.nll) else "N/A"
            if node.gpd_params is not None and len(node.gpd_params) == 2:
                s, g = node.gpd_params
                s_str = f"{s:.3f}" if abs(s) > 1e-4 else f"{s:.3e}"
                print(f"{prefix}, GPD(σ={s_str}, γ={g:.3f}), NLL={nll_str}")
            else:
                print(f"{prefix}, GPD(params=None), NLL={nll_str}")
        else:
            print(f"{prefix}, Unknown Node Type")
    else:
        if node.split_var is None or node.split_thr is None:
            print(f"{indent}Error: Internal node missing split info. {n_samples_str}")
            return
        if feat_names and 0 <= node.split_var < len(feat_names):
            feat = feat_names[node.split_var]
        else:
            feat = f"X{node.split_var}"
        # Pretty-print split condition (numeric vs categorical)
        if getattr(node, "split_cats", None) is not None:
            cond = f"{feat} ∈ {sorted(list(node.split_cats))}"
        else:
            cond = f"{feat} ≤ {node.split_thr:.6g}" if node.split_thr is not None else f"{feat} ?"
        if hasattr(node, 'gain') and node.gain > 0:  # L1/L2
            gain_info = f"(Gain={node.gain:.3f})"
        elif hasattr(node, 'split_gain'):
            gain_info = f"(PenalizedGain={node.split_gain:.3f})"
        else:
            gain_info = ""
        print(f"{indent}[{feat} ≤ {node.split_thr:.4g}] {gain_info} {n_samples_str}")
        if node.left:
            print_tree_structure(node.left, feat_names, indent + "  ")
        else:
            print(f"{indent}  (Left child missing or pruned)")
        if node.right:
            print_tree_structure(node.right, feat_names, indent + "  ")
        else:
            print(f"{indent}  (Right child missing or pruned)")


def validate_tree_structure(tree: NodeBase, min_samples_leaf: int = 40) -> bool:
    """Basic validation of tree structure."""
    nodes = [tree]
    while nodes:
        node = nodes.pop(0)
        if node is None or not hasattr(node, 'is_leaf'):
            logging.error("Validation Error: Encountered invalid node.")
            return False
        if node.is_leaf:
            if hasattr(node, 'n_samples') and node.n_samples > 0 and node.n_samples < min_samples_leaf:
                logging.warning(f"Validation Warning: Leaf has {node.n_samples} samples, less than min {min_samples_leaf}.")
        else:
            if node.left is None or node.right is None or node.split_var is None or node.split_thr is None:
                logging.error(f"Validation Error: Internal node at depth {node.depth} missing children or split info.")
                return False
            nodes.extend([node.left, node.right])
    return True

def normalization_constant(y_excess: np.ndarray, method: str = "1_over_k") -> float:
    """
    For pruning objectives like (mean NLL + alpha*K), this can be kept as 1.0.
    If you align with theory (1/k scaling), return 1/k.
    """
    k = int(np.sum(np.isfinite(y_excess)))
    if k <= 0:
        return 0.0
    if method == "1_over_k":
        return 1.0 / float(k)
    # Default: no scaling (useful when alpha absorbs constants)
    return 1.0


def theoretical_alpha_grid(
    k: int,
    n_points: int = 12,
    span: Tuple[float, float] = (0.25, 4.0)
) -> List[float]:
    """
    alpha_c = c * sqrt(log k) / sqrt(k), c in [span[0], span[1]]
    """
    if k <= 1:
        return [0.0]
    c_vals = np.linspace(span[0], span[1], n_points)
    base = np.sqrt(np.log(k)) / np.sqrt(k)
    return (c_vals * base).tolist()


# ---------------------------------------------------------------------
# 5) Leaf-wise 95% CI for gamma via numerical Hessian
# ---------------------------------------------------------------------
def leaf_confidence_intervals(
    tree: Any,
    X: np.ndarray,
    y_excess: np.ndarray,
    *,
    find_leaf: Callable[[Any, np.ndarray], Any],
    get_leaves: Callable[[Any], List[Any]],
    get_leaf_params: Callable[[Any], Tuple[float, float]],
    rel_eps: float = 1e-4,
    z_crit: float = 1.959963984540054  # ~ N(0,1) 97.5% quantile
) -> List[Dict[str, Any]]:
    """
    (패치 버전) leaf_id 없이 'leaf 객체 동일성'으로 매핑.
    각 리프의 MLE 근방에서 수치 Hessian을 계산해 Var(gamma)로 95% CI 산출.
    """
    X = np.asarray(X)
    y = np.asarray(y_excess, dtype=float)

    # 1) 전 샘플에 대해 소속 리프(객체)를 한 번에 매핑
    n = X.shape[0]
    leaf_objs = np.empty(n, dtype=object)
    for i in range(n):
        leaf_objs[i] = find_leaf(tree, X[i])

    # 2) 각 리프별로 y를 모아 CI 계산
    results: List[Dict[str, Any]] = []
    leaves = list(get_leaves(tree))

    for idx, leaf in enumerate(leaves):
        try:
            sigma_mle, gamma_mle = get_leaf_params(leaf)
        except Exception:
            results.append({"leaf_index": idx, "gamma": np.nan, "ci_95": [np.nan, np.nan]})
            continue

        # 동일 객체 비교로 해당 리프의 데이터 선택
        mask = (leaf_objs == leaf)
        y_leaf = y[mask]

        if (y_leaf.size < 3) or (not np.isfinite(sigma_mle)) or (not np.isfinite(gamma_mle)):
            results.append({"leaf_index": idx, "gamma": float(gamma_mle), "ci_95": [np.nan, np.nan]})
            continue

        # (σ,γ)에서 NLL의 수치 Hessian → 공분산 → gamma의 표준오차
        sigma0 = max(float(sigma_mle), 1e-6)
        theta0 = np.array([sigma0, float(gamma_mle)], dtype=float)

        def nll_theta(theta: np.ndarray) -> float:
            s, g = float(theta[0]), float(theta[1])
            return gpd_nll(y_leaf, s, g)  # gpd_nll는 이 파일에 이미 벡터화 구현됨

        H = _numerical_hessian_2d(nll_theta, theta0, rel_eps=rel_eps)
        V = _safe_inv_2x2(H)
        if (V is None) or (not np.isfinite(V[1, 1])):
            ci = [np.nan, np.nan]
        else:
            se_gamma = float(np.sqrt(max(V[1, 1], 0.0)))
            ci = [float(gamma_mle - z_crit * se_gamma), float(gamma_mle + z_crit * se_gamma)]

        results.append({"leaf_index": idx, "gamma": float(gamma_mle), "ci_95": ci})

    return results



# ---------------------------------------------------------------------
# 6) Permutation variable importance (delta mean NLL)
# ---------------------------------------------------------------------
def _score_mean_nll(
    tree: Any,
    X: np.ndarray,
    y_excess: np.ndarray,
    *,
    find_leaf: Callable[[Any, np.ndarray], Any] = find_leaf,
    get_leaf_params: Optional[Callable[[Any], Tuple[float, float]]] = None,
) -> float:
    """
    (패치) 리프별 배치화로 평균 NLL 계산. get_leaf_params가 주어지면 그것을,
    없으면 leaf.gpd_params를 사용.
    """
    X = np.asarray(X)
    y = np.asarray(y_excess, dtype=float)
    n = y.size
    if n == 0:
        return np.nan

    unique_ids, groups_idx, id2leaf, _ = _group_by_leaf_ids(tree, X, find_leaf=find_leaf)

    total = 0.0
    for grp_idx, oid in zip(groups_idx, unique_ids):
        lf = id2leaf.get(oid, None)
        if lf is None:
            return np.nan
        if get_leaf_params is not None:
            sigma, gamma = get_leaf_params(lf)
        else:
            gp = getattr(lf, "gpd_params", None)
            if gp is None:
                return np.nan
            sigma, gamma = float(gp[0]), float(gp[1])
        y_grp = y[grp_idx]
        nll = gpd_nll(y_grp, float(sigma), float(gamma))
        if not np.isfinite(nll):
            return np.nan
        total += nll

    return float(total / n)




def permutation_variable_importance(
    tree: Any,
    X: np.ndarray,
    y_excess: np.ndarray,
    feature_names: List[str],
    *,
    find_leaf: Callable[[Any, np.ndarray], Any] = find_leaf,
    get_leaf_params: Optional[Callable[[Any], Tuple[float, float]]] = None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> List[Dict[str, float]]:
    """
    (패치) 검증셋 GP NLL 증가분의 상대값( (NLL_perm - NLL_base) / NLL_base )을
    n_repeats 번 평균해 중요도를 계산.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    y = np.asarray(y_excess, dtype=float)

    base = get_subtree_nll(tree, X, y)
    base = float(base if np.isfinite(base) else np.inf)
    if not np.isfinite(base) or base <= 0:
        # base가 비정상이면 절대 증가값 사용(최후 방어)
        use_relative = False
    else:
        use_relative = True

    out = []
    for j, name in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            nll = get_subtree_nll(tree, Xp, y)
            if not np.isfinite(nll):
                continue
            delta = (nll - base)
            if use_relative:
                delta = delta / base
            deltas.append(delta)
        importance = float(np.mean(deltas)) if deltas else 0.0
        out.append({"name": name, "importance": importance})

    out.sort(key=lambda d: d["importance"], reverse=True)
    return out

    
    
def _group_by_leaf_ids(tree: Any, X: np.ndarray, *, find_leaf: Callable[[Any, np.ndarray], Any]):
    """
    각 샘플의 소속 리프(객체)를 구해, leaf id 순으로 정렬·그룹핑한다.
    반환:
      unique_ids: 고유 leaf id 배열 (정렬됨)
      groups_idx: 각 leaf에 해당하는 X 인덱스 배열 리스트
      id2leaf: id -> leaf 객체 매핑 dict
      leaf_objs: 샘플별 leaf 객체 배열
    """
    n = X.shape[0]
    leaf_objs = np.empty(n, dtype=object)
    for i in range(n):
        leaf_objs[i] = find_leaf(tree, X[i])

    ids = np.fromiter((id(obj) for obj in leaf_objs), dtype=np.int64, count=n)
    order = np.argsort(ids)
    ids_sorted = ids[order]

    boundaries = np.where(np.diff(ids_sorted) != 0)[0] + 1
    groups_idx = np.split(order, boundaries)
    unique_ids = np.unique(ids_sorted)

    id2leaf = {}
    seen = set()
    for obj in leaf_objs:
        oid = id(obj)
        if oid not in seen:
            id2leaf[oid] = obj
            seen.add(oid)

    return unique_ids, groups_idx, id2leaf, leaf_objs

# ---------------------------------------------------------------------#
# 7. Data Preprocessing Utility
# ---------------------------------------------------------------------#

def create_lagged_features(df: pd.DataFrame,
                           target_col: str,
                           lag_config: Dict[str, List[int]],
                           date_col: str,
                           start_year: Optional[int] = None,
                           additional_features: Optional[List[str]] = None
                           ) -> Tuple[pd.DataFrame, List[str]]:
    """
    Creates lagged and (auto) differenced features.

    Rules:
      - If key ends with '_diff' => attempt log-diff.
      - If any non-positive values exist in series => fallback to simple diff.
    """
    df_proc = df.copy()

    # Date handling
    if not pd.api.types.is_datetime64_any_dtype(df_proc[date_col]):
        df_proc[date_col] = pd.to_datetime(df_proc[date_col], errors='coerce')
    df_proc = df_proc.sort_values(by=date_col).reset_index(drop=True)

    lagged_feature_names: List[str] = []
    for var, lags in lag_config.items():
        if var.endswith("_diff"):
            base_var = var.removesuffix("_diff")
            if base_var not in df_proc.columns:
                logging.warning(f"Column '{base_var}' for differencing not found. Skipping.")
                continue
            s = df_proc[base_var].astype(float)
            if (s > 0).all():
                diff_series = np.log(s).diff()
            else:
                logging.info(f"'{base_var}' has non-positive values; using simple differencing instead of log-diff.")
                diff_series = s.diff()
            for lag in lags:
                if lag <= 0:
                    continue
                lagged_col_name = f"{var}_lag{lag}"
                df_proc[lagged_col_name] = diff_series.shift(lag)
                lagged_feature_names.append(lagged_col_name)
        else:
            if var not in df_proc.columns:
                logging.warning(f"Column '{var}' for lagging not found. Skipping.")
                continue
            for lag in lags:
                if lag <= 0:
                    continue
                lagged_col_name = f"{var}_lag{lag}"
                df_proc[lagged_col_name] = df_proc[var].shift(lag)
                lagged_feature_names.append(lagged_col_name)

    all_feature_names = lagged_feature_names + (additional_features or [])
    final_cols = [target_col] + list(dict.fromkeys(all_feature_names))  # unique, keep order

    if start_year is not None:
        df_proc = df_proc[df_proc[date_col].dt.year >= start_year].copy()

    final_df = df_proc[final_cols].dropna().reset_index(drop=True)
    feature_names_used = [col for col in final_df.columns if col != target_col]

    logging.info(f"Created dataset with {len(final_df)} samples and {len(feature_names_used)} features.")
    return final_df, feature_names_used
