import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
from scipy import stats
import copy
from typing import Tuple, List, Optional, Dict, Any, Union
import math

# ---------------------------------------------------------------------#
# Logging Setup
# ---------------------------------------------------------------------#
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# ---------------------------------------------------------------------#
# Constants
# ---------------------------------------------------------------------#
# SEED = 42
# np.random.seed(SEED)
# MIN_FLOAT = 1e-9 # Small number to avoid division by zero etc.
# LARGE_FLOAT = 1e15 # Large number for infinite costs/NLL

FLT_INFO = np.finfo(float)

SEED        = 42
rng         = np.random.default_rng(SEED)     # 전역 RandomGenerator
MIN_FLOAT   = FLT_INFO.eps                    # ≈2.22e‑16
LARGE_FLOAT = 1e30

# ---------------------------------------------------------------------#
# 0. Utility Functions
# ---------------------------------------------------------------------#
def mse_loss(y: np.ndarray, mu: float) -> float:
    """Calculate Mean Squared Error."""
    if len(y) == 0:
        return 0.0
    return np.sum((y - mu) ** 2)

def mae_loss(y: np.ndarray, md: float) -> float:
    """Calculate Mean Absolute Error."""
    if len(y) == 0:
        return 0.0
    return np.sum(np.abs(y - md))

# ---------------------------------------------------------------------#
# Base Node Class and Specific Nodes
# ---------------------------------------------------------------------#
class NodeBase:
    """Base class for all nodes in the trees."""
    __slots__ = ("is_leaf", "split_var", "split_thr", "gain", "left", "right", "depth", "n_samples")

    def __init__(self):
        self.is_leaf: bool = True
        self.split_var: Optional[int] = None
        self.split_thr: Optional[float] = None
        self.gain: float = 0.0
        self.left: Optional['NodeBase'] = None
        self.right: Optional['NodeBase'] = None
        self.depth: int = 0
        self.n_samples: int = 0 # Keep track of samples in the node

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
    __slots__ = ("gpd_params", "nll", "split_gain") # Inherits others via super().__init__()

    def __init__(self):
        super().__init__() # Initializes is_leaf, split_var, split_thr, left, right, depth, n_samples
        # Note: 'gain' from NodeBase is repurposed as 'split_gain' here for clarity in GPD context
        self.split_gain: float = 0.0 # Gain used for splitting decision (AIC penalized)
        self.gpd_params: Optional[Tuple[float, float]] = None # (sigma, gamma)
        self.nll: Optional[float] = None # Negative Log-Likelihood at this node



def _nll_lognormal(params: Tuple[float, float],
                   y: np.ndarray,
                   *, trunc_left: float = 0.0) -> float:
    """
    Negative Log‑Likelihood for (optionally *left‑truncated*) LogNormal.
    trunc_left = 0  → 일반 LogNormal
    trunc_left = u  → P(Y>u) 로 조건부화한 LogNormal
    """
    mu, sigma = params
    if sigma <= MIN_FLOAT:
        return LARGE_FLOAT

    y_pos = y[y > 0]
    if y_pos.size == 0:
        return LARGE_FLOAT

    log_y = np.log(y_pos)
    z     = (log_y - mu) / sigma
    base  = np.sum(np.log(sigma) + 0.5*np.log(2*np.pi) + 0.5*z**2 + log_y)

    if trunc_left > 0:
        # S = survival func  = P(Y>u) = 1 - Φ((ln u - μ)/σ)
        zc   = (np.log(trunc_left) - mu) / sigma
        surv = 1.0 - 0.5*(1 + math.erf(zc/np.sqrt(2)))
        if surv <= 0:
            return LARGE_FLOAT
        base -= y_pos.size * np.log(surv)

    return base if np.isfinite(base) else LARGE_FLOAT



def fit_lognormal_mle(y: np.ndarray, *, trunc_left: float = 0.0
                      ) -> Tuple[float, float]:
    """MLE for (optionally left‑truncated) LogNormal → (μ̂, σ̂)."""
    y_pos = y[y > 0]
    n_pos = y_pos.size
    if n_pos < 2:
        if n_pos == 1:
            return float(np.log(y_pos[0])), MIN_FLOAT*10
        return 0.0, 1.0

    log_y = np.log(y_pos)
    init   = [log_y.mean(), max(log_y.std(), MIN_FLOAT*10)]
    bounds = [(None, None), (MIN_FLOAT*10, None)]

    obj = lambda p: _nll_lognormal(p, y_pos, trunc_left=trunc_left)

    try:
        res = minimize(obj, init, method="L-BFGS-B", bounds=bounds)
        if res.success and np.all(np.isfinite(res.x)):
            μ, σ = res.x
            σ = max(σ, MIN_FLOAT*100)
            return float(μ), float(σ)
    except Exception as e:
        logging.debug(f"LogNormal MLE failed: {e}")

    return tuple(init)


# ---------------------------------------------------------------------#
# GPD MLE and Cost Function
# ---------------------------------------------------------------------#
def neg_loglik_gpd(theta: Tuple[float, float], y: np.ndarray) -> float:
    """
    Calculate negative log-likelihood for Generalized Pareto Distribution (GPD).
    Handles the case gamma -> 0 (Exponential distribution limit).
    """
    sigma, gamma = theta
    n = len(y)
    if n == 0:
        return 0.0
    if sigma <= MIN_FLOAT: # Sigma must be positive
        return LARGE_FLOAT

    # Handle Exponential distribution case (gamma ≈ 0)
    if abs(gamma) < 1e-6:
        if np.any(y < 0): # Exponential is for non-negative values
             return LARGE_FLOAT
        nll = n * np.log(sigma) + np.sum(y) / sigma
    else:
        # Standard GPD case
        z = 1 + gamma * y / sigma
        if np.any(z <= MIN_FLOAT): # Argument of log must be positive
            return LARGE_FLOAT
        # Correct GPD log-likelihood: sum(log(sigma) + (1/gamma + 1) * log(z))
        nll = n * np.log(sigma) + (1 / gamma + 1) * np.sum(np.log(z))

    # Check for non-finite results which can break optimization
    if not np.isfinite(nll):
        return LARGE_FLOAT
    return nll
def fit_gpd_mle(y: np.ndarray,
                *, gamma_bounds: Tuple[float, float] = (-0.5, 0.95)) -> np.ndarray:
    """
    Stabilized GPD MLE fitting.
    - Returns [sigma, gamma].
    - Uses PWM for initialization.
    - Applies boundary penalties.
    - Enforces sigma lower bound.
    - Handles small samples.
    """
    bounds = [(MIN_FLOAT*10, None), gamma_bounds]
    n = len(y)
    # 0) Handle very small samples
    if n < 30:
        # Fallback: Use median for scale, slightly negative shape
        # Ensure sigma is positive
        fallback_sigma = max(np.median(y), MIN_FLOAT * 10) if n > 0 else 1.0
        return np.array([fallback_sigma, -0.2])

    # 1) Probability-Weighted Moments (PWM) initial estimate
    y_sorted = np.sort(y)
    # Use unbiased PWM estimators if available, otherwise simple moments
    # For simplicity here, use sample mean and median heuristic
    y_bar = np.mean(y)
    y_med = np.median(y)
    # Heuristic initial gamma (can be unstable, hence clipping)
    gamma0_raw = (y_bar - 2.0 * y_med) / (y_bar - y_med + MIN_FLOAT)
    gamma0 = np.clip(gamma0_raw, -0.4, 0.9) # Clip to reasonable range
    # Initial sigma based on gamma0 and mean, ensuring positivity
    sigma0 = max(abs(gamma0) * y_bar if abs(gamma0) > 1e-4 else y_bar, MIN_FLOAT * 10) # Use y_bar if gamma near 0

    init_params = np.array([sigma0, gamma0])

    # 2) Optimization settings
    #bounds = [(MIN_FLOAT * 10, None), (-0.5, 0.95)] # Sigma > 0, Gamma constrained

    def objective_with_penalty(theta: np.ndarray, data: np.ndarray) -> float:
        """Objective function with boundary penalties for GPD NLL."""
        s, g = theta
        base_nll = neg_loglik_gpd(theta, data)
        penalty = 0.0
        # Penalize being too close to boundaries where optimization might struggle
        if abs(g) < 1e-4 or abs(g + 0.5) < 1e-3 or abs(g - 0.95) < 1e-3:
            penalty = 1e2 # Fixed penalty, could be adaptive

        # Penalize non-finite NLL heavily
        if not np.isfinite(base_nll):
             return LARGE_FLOAT

        return base_nll + penalty

    # 3) Run L-BFGS-B optimizer
    try:
        res = minimize(
            objective_with_penalty,
            x0=init_params,
            args=(y,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-9, "eps": 1e-7}, # Added eps for numerical differentiation
        )

        # 4) Check success and enforce sigma lower bound
        if res.success and np.isfinite(res.fun):
            s_opt, g_opt = res.x
            # Ensure sigma meets the minimum requirement
            final_sigma = max(s_opt, MIN_FLOAT * 10)
            return np.array([final_sigma, g_opt])
        else:
             logging.debug(f"GPD MLE optimization failed (Status: {res.status}, Msg: {res.message}). Returning PWM init.")
             # Fallback to initial estimate if optimization failed
             return init_params

    except (ValueError, RuntimeWarning, Exception) as e:
        logging.debug(f"GPD MLE failed with error: {e}. Returning PWM init: {init_params}")
        # Fallback to initial estimate on any exception
        return init_params

def node_cost_gpd(y: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Calculate node cost (NLL) and parameters for GPD.
    Returns (cost, [sigma, gamma]). Handles failures gracefully.
    """
    if len(y) == 0:
        return 0.0, np.array([1.0, 0.0]) # Default params for empty node

    try:
        theta = fit_gpd_mle(y)
        cost = neg_loglik_gpd(theta, y)

        # If cost is non-finite, return a large cost and the estimated (possibly bad) theta
        if not np.isfinite(cost):
            logging.debug(f"Non-finite GPD cost ({cost}) for theta={theta}. Returning large cost.")
            return LARGE_FLOAT, theta
        return cost, theta
    except Exception as e:
        # Broad exception catch during cost calculation (should be rare if fit_gpd_mle is robust)
        logging.warning(f"Error during node_cost_gpd: {e}. Returning large cost.")
        # Provide default parameters in case of complete failure
        fallback_sigma = max(np.median(y), MIN_FLOAT * 10) if len(y) > 0 else 1.0
        fallback_theta = np.array([fallback_sigma, -0.2])
        return LARGE_FLOAT, fallback_theta




# ---------------------------------------------------------------------#
# 2. CART Splitting Functions
# ---------------------------------------------------------------------#
def _best_split_l1(x: np.ndarray, y: np.ndarray, min_leaf: int = 1
                   ) -> Tuple[float, Optional[float]]:
    """Find the best split point for a node in L1 CART (median-based)."""
    n = len(y)
    if n < 2 * min_leaf: # Cannot split if not enough samples
        return 0.0, None

    # Sort data by the feature x
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]

    parent_median = np.median(ys)
    parent_loss = mae_loss(ys, parent_median)
    best_gain, best_thr = 0.0, None

    # Iterate through possible split points
    # A split occurs between i-1 and i
    for i in range(min_leaf, n - min_leaf ):
        # Only consider splits where x value changes
        if xs[i] == xs[i - 1]:
            continue

        # Calculate loss for left and right children
        left_y, right_y = ys[:i], ys[i:]
        median_l = np.median(left_y)
        median_r = np.median(right_y)
        child_loss = mae_loss(left_y, median_l) + mae_loss(right_y, median_r)

        # Calculate gain
        gain = parent_loss - child_loss

        # Update best split if current gain is higher
        if gain > best_gain:
            best_gain = gain
            # Threshold is midpoint between the two x values
            best_thr = 0.5 * (xs[i] + xs[i - 1])

    # Avoid negligible gains
    if best_gain < MIN_FLOAT:
        return 0.0, None

    return best_gain, best_thr

def _best_split_l2(x: np.ndarray, y: np.ndarray, min_leaf: int = 1
                   ) -> Tuple[float, Optional[float]]:
    """Find the best split point for a node in L2 CART (mean-based) efficiently."""
    n = len(y)
    if n < 2 * min_leaf:
        return 0.0, None

    # Sort data by the feature x
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]

    # Calculate parent loss (variance * n)
    parent_mean = np.mean(ys)
    parent_loss = mse_loss(ys, parent_mean) # Equivalent to np.sum((ys - parent_mean)**2)

    # Precompute cumulative sums for efficiency
    csum = np.cumsum(ys)
    csq = np.cumsum(ys ** 2)
    total_sum, total_sq = csum[-1], csq[-1]

    best_gain, best_thr = 0.0, None

    # Iterate through possible split points (between i-1 and i)
    for i in range(min_leaf, n - min_leaf): # Ensure children have at least min_leaf samples
        if xs[i] == xs[i - 1]: # Skip if x value hasn't changed
            continue

        # Left child calculations
        sum_l = csum[i - 1]
        sum_sq_l = csq[i - 1]
        cnt_l = i
        mean_l = sum_l / cnt_l
        # Loss_l = sum(y_l^2) - n_l * mean_l^2 (more stable form of sum((y_l-mean_l)^2))
        loss_l = sum_sq_l - cnt_l * (mean_l ** 2)

        # Right child calculations
        sum_r = total_sum - sum_l
        sum_sq_r = total_sq - sum_sq_l
        cnt_r = n - cnt_l
        mean_r = sum_r / cnt_r
        loss_r = sum_sq_r - cnt_r * (mean_r ** 2)

        # Total loss of children
        child_loss = loss_l + loss_r

        # Gain is reduction in impurity (sum of squares)
        gain = parent_loss - child_loss

        # Update best split
        if gain > best_gain:
            best_gain = gain
            best_thr = 0.5 * (xs[i] + xs[i - 1])

    # Avoid negligible gains
    if best_gain < MIN_FLOAT:
         return 0.0, None

    return best_gain, best_thr

def _best_split_gpd(x: np.ndarray, y: np.ndarray, min_leaf: int = 30,
                    penalty: float = 2.0) -> Tuple[float, Optional[float]]:
    """
    Finds the best split point for a node in GPD-CART using NLL reduction.
    - Uses AIC-like penalty for complexity.
    - Returns (penalized_gain, threshold).
    """
    n = len(y)
    if n < 2 * min_leaf: # Cannot split
        return 0.0, None

    # Sort by feature x
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]

    # Calculate parent node NLL (cost)
    parent_nll, _ = node_cost_gpd(ys)
    if not np.isfinite(parent_nll): # Cannot split if parent cost is invalid
         logging.warning("Parent GPD NLL is not finite. Cannot split.")
         return 0.0, None

    best_gain, best_thr = -LARGE_FLOAT, None # Initialize gain to very small value

    # Iterate through potential split points
    for i in range(min_leaf, n - min_leaf): # Ensure children meet min_leaf size
        if xs[i] == xs[i - 1]: # Skip if x value is the same
            continue

        left_y, right_y = ys[:i], ys[i:]

        # Calculate NLL for potential children
        l_nll, _ = node_cost_gpd(left_y)
        r_nll, _ = node_cost_gpd(right_y)

        # Check if child NLLs are valid
        if not (np.isfinite(l_nll) and np.isfinite(r_nll)):
            continue # Skip split if children fitting failed

        # Calculate AIC-penalized gain: Gain = ParentNLL - ChildNLL_Sum - Penalty
        # Higher gain is better (means larger reduction in NLL relative to penalty)
        gain = parent_nll - (l_nll + r_nll) - penalty*4

        if gain > best_gain:
            best_gain = gain
            best_thr = 0.5 * (xs[i] + xs[i - 1])

    # Only return a split if the penalized gain is positive
    if best_gain <= 0: # Changed from < MIN_FLOAT to <= 0 for AIC penalty logic
        return 0.0, None

    return best_gain, best_thr


# ---------------------------------------------------------------------#
# 3. Tree Growing Functions
# ---------------------------------------------------------------------#
def grow_tree_l1(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                 min_leaf: int = 20, max_depth: int = 3, depth: int = 0) -> NodeL1:
    """Grows an L1 CART (median-based) recursively."""
    n_samples, n_features = X.shape
    node = NodeL1()
    node.depth = depth
    node.n_samples = n_samples
    node.median_val = np.median(y) if n_samples > 0 else 0.0

    # Stopping conditions
    if n_samples < 2 * min_leaf or depth >= max_depth or n_samples == 0:
        return node # Return leaf node

    best_gain, best_var, best_thr = -1.0, None, None # Initialize gain to negative

    # Find the best split across all features
    for j in range(n_features):
        gain, thr = _best_split_l1(X[:, j], y, min_leaf)
        if thr is not None and gain > best_gain:
            best_gain = gain
            best_var = j
            best_thr = thr

    # If no beneficial split is found (gain <= 0 or no threshold)
    if best_thr is None or best_gain <= MIN_FLOAT:
        return node # Return leaf node

    # Ensure the split actually divides the data and meets min_leaf requirement
    mask_left = X[:, best_var] <= best_thr
    n_left = np.sum(mask_left)
    n_right = n_samples - n_left

    if n_left < min_leaf or n_right < min_leaf:
        return node # Split doesn't meet min_leaf criteria, make it a leaf

    # Create internal node
    node.is_leaf = False
    node.split_var = best_var
    node.split_thr = best_thr
    node.gain = best_gain

    # Recursively grow children
    logging.debug(f"Depth {depth}: Splitting on '{feature_names[best_var]}' <= {best_thr:.3g}, Gain={best_gain:.3f}, N={n_samples}->({n_left}, {n_right})")
    node.left = grow_tree_l1(X[mask_left], y[mask_left], feature_names, min_leaf, max_depth, depth + 1)
    node.right = grow_tree_l1(X[~mask_left], y[~mask_left], feature_names, min_leaf, max_depth, depth + 1)

    return node

def grow_tree_l2(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                 min_leaf: int = 30, max_depth: int = 4, depth: int = 0) -> NodeL2:
    """Grows an L2 CART (mean-based) recursively."""
    n_samples, n_features = X.shape
    node = NodeL2()
    node.depth = depth
    node.n_samples = n_samples
    node.mean_val = np.mean(y) if n_samples > 0 else 0.0

    # Stopping conditions
    if n_samples < 2 * min_leaf or depth >= max_depth or n_samples == 0:
        return node

    best_gain, best_var, best_thr = -1.0, None, None

    # Find best split
    for j in range(n_features):
        gain, thr = _best_split_l2(X[:, j], y, min_leaf)
        if thr is not None and gain > best_gain:
            best_gain = gain
            best_var = j
            best_thr = thr

    # Check if a valid split was found
    if best_thr is None or best_gain <= MIN_FLOAT:
        return node

    # Verify min_leaf constraint for children
    mask_left = X[:, best_var] <= best_thr
    n_left = np.sum(mask_left)
    n_right = n_samples - n_left

    if n_left < min_leaf or n_right < min_leaf:
        return node

    # Create internal node
    node.is_leaf = False
    node.split_var = best_var
    node.split_thr = best_thr
    node.gain = best_gain

    # Recursive calls
    logging.debug(f"Depth {depth}: Splitting on '{feature_names[best_var]}' <= {best_thr:.3g}, Gain={best_gain:.3f}, N={n_samples}->({n_left}, {n_right})")
    node.left = grow_tree_l2(X[mask_left], y[mask_left], feature_names, min_leaf, max_depth, depth + 1)
    node.right = grow_tree_l2(X[~mask_left], y[~mask_left], feature_names, min_leaf, max_depth, depth + 1)

    return node

def grow_tree_gpd(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                  min_leaf: int = 100, max_depth: int = 3, depth: int = 0) -> NodeGPD:
    """Grows a GPD CART recursively."""
    n_samples, n_features = X.shape
    node = NodeGPD()
    node.depth = depth
    node.n_samples = n_samples

    # Calculate GPD parameters and NLL for the current node
    node.nll, node.gpd_params = node_cost_gpd(y)

    # Stopping conditions
    if n_samples < 2 * min_leaf or depth >= max_depth or n_samples == 0:
         if node.gpd_params is None: # Ensure params exist even for small leaves
              node.nll, node.gpd_params = node_cost_gpd(y) # Try one last time
         logging.debug(f"Depth {depth}: Leaf created (Stop condition). N={n_samples}, NLL={node.nll:.2f}")
         return node

    if node.gpd_params is None or not np.isfinite(node.nll):
         logging.warning(f"Depth {depth}: Leaf created (GPD fit failed). N={n_samples}")
         return node # Cannot split if current node fit failed

    best_gain, best_var, best_thr = -LARGE_FLOAT, None, None

    # Find best split based on penalized NLL reduction
    for j in range(n_features):
        # Using default penalty=2.0 from _best_split_gpd definition
        gain, thr = _best_split_gpd(X[:, j], y, min_leaf)
        if thr is not None and gain > best_gain:
            best_gain = gain
            best_var = j
            best_thr = thr

    # If no split provides positive penalized gain
    if best_thr is None or best_gain <= 0: # Use <= 0 for penalized gain
        logging.debug(f"Depth {depth}: Leaf created (No positive gain split). N={n_samples}, NLL={node.nll:.2f}")
        return node

    # Verify min_leaf constraint for the chosen split
    mask_left = X[:, best_var] <= best_thr
    n_left = np.sum(mask_left)
    n_right = n_samples - n_left

    if n_left < min_leaf or n_right < min_leaf:
        logging.debug(f"Depth {depth}: Leaf created (Split violates min_leaf). N={n_samples}, NLL={node.nll:.2f}")
        return node

    # Create internal node
    node.is_leaf = False
    node.split_var = best_var
    node.split_thr = best_thr
    node.split_gain = best_gain # Store the penalized gain

    # Recursive calls
    logging.debug(f"Depth {depth}: Splitting on '{feature_names[best_var]}' <= {best_thr:.3g}, Penalized Gain={best_gain:.3f}, N={n_samples}->({n_left}, {n_right})")
    node.left = grow_tree_gpd(X[mask_left], y[mask_left], feature_names, min_leaf, max_depth, depth + 1)
    node.right = grow_tree_gpd(X[~mask_left], y[~mask_left], feature_names, min_leaf, max_depth, depth + 1)

    return node

# ---------------------------------------------------------------------#
# 4. Leaf Finding and Parameter Assignment
# ---------------------------------------------------------------------#
def find_leaf(node: Union[NodeL1, NodeL2, NodeGPD], x: np.ndarray) -> Union[NodeL1, NodeL2, NodeGPD]:
    """Finds the leaf node corresponding to a single data point x."""
    current_node = node
    while not current_node.is_leaf:
        if x[current_node.split_var] <= current_node.split_thr:
            current_node = current_node.left
        else:
            current_node = current_node.right
    return current_node

def _get_all_leaves(node: Union[NodeL1, NodeL2, NodeGPD]) -> List[Union[NodeL1, NodeL2, NodeGPD]]:
    """Helper to recursively collect all leaf nodes."""
    leaves = []
    if node.is_leaf:
        leaves.append(node)
    else:
        if node.left:
            leaves.extend(_get_all_leaves(node.left))
        if node.right:
            leaves.extend(_get_all_leaves(node.right))
    return leaves
def assign_lognorm_params(root: Union[NodeL1, NodeL2],
                          X: np.ndarray, y: np.ndarray,
                          *, trunc_left: float = 0.0) -> None:
    """Fit LN(μ,σ) per leaf; supports left truncation."""
    if root is None:
        return

    leaves = _get_all_leaves(root)
    leaf_id = {id(l): i for i, l in enumerate(leaves)}
    leaf_idx = np.array([leaf_id[id(find_leaf(root, x))] for x in X])

    for i, lf in enumerate(leaves):
        y_leaf = y[leaf_idx == i]
        lf.n_samples = y_leaf.size
        if y_leaf.size:
            μ, σ = fit_lognormal_mle(y_leaf, trunc_left=trunc_left)
            lf.lognorm_mu, lf.lognorm_sigma = μ, σ
        else:       # 빈 leaf 방어
            lf.lognorm_mu, lf.lognorm_sigma = 0.0, 1.0
            if isinstance(lf, NodeL1): lf.median_val = 0.0
            if isinstance(lf, NodeL2): lf.mean_val   = 0.0


# Note: assign_lognorm_l1 and assign_lognorm_l2 are now covered by assign_lognorm_params

# ---------------------------------------------------------------------#
# 5. GPD Tree Pruning (Cost-Complexity Pruning with CV)
# ---------------------------------------------------------------------#

def get_subtree_nll(node: NodeGPD, X: np.ndarray, y: np.ndarray) -> float:
    """Calculates the total NLL of data (X, y) evaluated on a GPD subtree."""
    total_nll = 0.0
    n_errors = 0
    if len(X) == 0:
        return 0.0

    for i in range(len(X)):
        xi, yi = X[i], y[i]
        leaf_node = find_leaf(node, xi)

        if leaf_node.gpd_params is None:
            # This shouldn't happen if tree was built correctly, but handle defensively
            total_nll += LARGE_FLOAT
            n_errors += 1
            continue

        sigma, gamma = leaf_node.gpd_params

        # Calculate NLL contribution for this point using the leaf's parameters
        # Re-use neg_loglik_gpd but for a single point (pass y as array of size 1)
        point_nll = neg_loglik_gpd((sigma, gamma), np.array([yi]))

        if not np.isfinite(point_nll):
            total_nll += LARGE_FLOAT # Penalize if point is outside support of fitted GPD
            n_errors += 1
        else:
            total_nll += point_nll

    if n_errors > 0:
         logging.debug(f"Encountered {n_errors}/{len(X)} errors (e.g., y outside GPD support) during NLL calculation.")

    return total_nll

def _get_pruning_sequence(node: NodeGPD) -> List[Tuple[float, NodeGPD]]:
    """Generates potential alpha values and weakest links for pruning."""
    # This computes the g(t) = (R(t) - R(Tt)) / (|Tt| - 1) term for each internal node t
    # R(t) is the NLL at node t, R(Tt) is the sum of NLLs of its descendant leaves
    # |Tt| is the number of leaves in the subtree rooted at t
    sequence = []

    def cost_and_leaves(n: NodeGPD):
        if n.is_leaf:
            return n.nll, 1
        else:
            left_nll_sum, left_leaves = cost_and_leaves(n.left)
            right_nll_sum, right_leaves = cost_and_leaves(n.right)
            total_leaf_nll = left_nll_sum + right_nll_sum
            total_leaves = left_leaves + right_leaves

            # Calculate alpha for this node
            # Avoid division by zero if subtree has only 1 leaf (shouldn't happen for internal)
            if total_leaves > 1 and np.isfinite(n.nll) and np.isfinite(total_leaf_nll):
                 alpha = (n.nll - total_leaf_nll) / (total_leaves - 1)
                 # Only consider non-negative alphas where pruning might improve cost-complexity
                 if alpha >= -MIN_FLOAT: # Allow alpha=0
                      sequence.append((alpha, n)) # Store alpha and the node itself
            else:
                 # If NLLs are invalid or only 1 leaf, assign very large alpha to prevent pruning here
                 sequence.append((LARGE_FLOAT, n))


            return total_leaf_nll, total_leaves

    if not node.is_leaf:
        cost_and_leaves(node)
        # Sort sequence by alpha (weakest links first)
        sequence.sort(key=lambda item: item[0])

    return sequence


def prune_gpd_single_alpha(root: NodeGPD,
                           alpha: float,
                           X: np.ndarray,
                           y: np.ndarray) -> NodeGPD:
    pruned = copy.deepcopy(root)
    def prune_node(node, X_sub, y_sub):
        if node.is_leaf:
            return node
        mask_left = X_sub[:, node.split_var] <= node.split_thr
        node.left  = prune_node(node.left,  X_sub[mask_left],  y_sub[mask_left])
        node.right = prune_node(node.right, X_sub[~mask_left], y_sub[~mask_left])
        if node.left.is_leaf and node.right.is_leaf:
            cost_int = node.left.nll + node.right.nll + alpha
            cost_leaf, theta_leaf = node_cost_gpd(y_sub)
            if cost_leaf <= cost_int:
                node.is_leaf = True
                node.left = node.right = None
                node.split_var = node.split_thr = None
                node.split_gain = 0.0
                node.nll, node.gpd_params = cost_leaf, theta_leaf
        return node
    return prune_node(pruned, X, y)



def prune_gpd_with_cv(root: NodeGPD, X: np.ndarray, y: np.ndarray,
                      n_folds: int = 5, random_state: int = SEED
                      ) -> Tuple[NodeGPD, float]:
    """
    Prunes the GPD tree using cost-complexity pruning with cross-validation
    to find the optimal alpha. Returns the pruned tree and the best alpha.
    (Simplified 1-pass CV approach).
    """
    n_samples = len(y)
    if n_samples < n_folds * 2 or n_samples < 30: # Need enough data for CV
        logging.warning("Dataset too small for reliable CV pruning. Returning unpruned tree.")
        return root, 0.0 # Return original tree with alpha=0

    # 1. Generate sequence of alphas from the full tree
    pruning_sequence = _get_pruning_sequence(root)
    alphas = sorted(list(set([alpha for alpha, node in pruning_sequence if alpha < LARGE_FLOAT])))
    # Include 0 alpha (unpruned tree relative to calculated alphas) and add a slightly larger value
    candidate_alphas = [0.0] + alphas
    if alphas:
         candidate_alphas.append(alphas[-1] * 1.1 + MIN_FLOAT) # Test alpha slightly larger than max found

    logging.info(f"Generated {len(candidate_alphas)} candidate alphas for CV pruning.")

    # 2. Prepare for Cross-Validation
    # Stratification might be useful if y distribution is highly skewed
    # Simple stratification based on median for demonstration
    try:
        y_stratify = (y > np.median(y)).astype(int)
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        folds = list(kf.split(X, y_stratify))
    except ValueError: # Handle cases where stratification fails (e.g., < 2 members in a class)
         logging.warning("Stratified KFold failed, using standard KFold.")
         kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
         folds = list(kf.split(X))


    # 3. Evaluate each alpha using CV
    cv_scores = {alpha: [] for alpha in candidate_alphas}

    for fold_idx, (train_indices, test_indices) in enumerate(folds):
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        if len(y_train) < 20 or len(y_test) == 0: # Skip fold if sets are too small
            logging.warning(f"Skipping CV fold {fold_idx+1} due to small train/test size.")
            continue

        # Build a tree on the training fold *This part is complex if we rebuild for each fold*
        # SIMPLIFICATION for 1-pass: Use the original full tree structure, but prune based on training data NLLs?
        # Or: Prune the original tree and evaluate on test set. Let's use the latter (more common).

        # For each candidate alpha, prune the *original* full tree
        for alpha in candidate_alphas:
            # Prune the original tree using this alpha
            # Need a function that prunes based on the NLL values already in the tree
            pruned_for_alpha = prune_gpd_single_alpha(root, alpha, X_train, y_train)

            # Evaluate this pruned tree on the test set of this fold
            test_nll = get_subtree_nll(pruned_for_alpha, X_test, y_test)
            if np.isfinite(test_nll):
                 cv_scores[alpha].append(test_nll / len(y_test)) # Store average NLL per sample
            else:
                 cv_scores[alpha].append(LARGE_FLOAT) # Penalize failures


    # 4. Select the best alpha
    mean_cv_scores = {alpha: np.mean(scores) for alpha, scores in cv_scores.items() if scores}

    if not mean_cv_scores: # Handle case where all folds failed
         logging.error("CV pruning failed for all alphas. Returning unpruned tree.")
         return root, 0.0

    # Find alpha with the minimum average NLL across folds
    best_alpha = min(mean_cv_scores, key=mean_cv_scores.get)
    min_score = mean_cv_scores[best_alpha]

    logging.info(f"CV Results: Best alpha = {best_alpha:.4g} with mean NLL = {min_score:.4f}")

    # 5. Prune the original tree using the best alpha found
    final_pruned_tree = prune_gpd_single_alpha(root, best_alpha, X, y)

    return final_pruned_tree, best_alpha

# prune_with_cv_once -> Renamed to prune_gpd_with_cv, implementation improved

# ---------------------------------------------------------------------#
# 6. Tree Visualization and Evaluation
# ---------------------------------------------------------------------#
def print_tree_structure(node: Union[NodeL1, NodeL2, NodeGPD],
                         feat_names: List[str], indent: str = ""):
    """모든 지원되는 트리 타입의 구조를 출력합니다."""
    if not hasattr(node, 'is_leaf'):
         print(f"{indent}오류: 유효하지 않은 노드 객체입니다.")
         return

    if node.is_leaf:
        # n_samples 속성이 있는지 확인 (NodeBase에 추가됨)
        n_samples_str = f"N={node.n_samples}" if hasattr(node, 'n_samples') else "N=?"
        prefix = f"{indent}Leaf: {n_samples_str}"

        if isinstance(node, NodeL1):
            # None 값 처리 추가
            median_str = f"{node.median_val:.2f}" if node.median_val is not None else "N/A"
            mu_str = f"{node.lognorm_mu:.2f}" if node.lognorm_mu is not None else "N/A"
            sigma_str = f"{node.lognorm_sigma:.2f}" if node.lognorm_sigma is not None else "N/A"
            print(f"{prefix}, median={median_str}, LN(mu={mu_str}, sigma={sigma_str})")
        elif isinstance(node, NodeL2):
            # None 값 처리 및 sigma 형식화 개선
            mean_str = f"{node.mean_val:.2f}" if node.mean_val is not None else "N/A"
            mu_str = f"{node.lognorm_mu:.2f}" if node.lognorm_mu is not None else "N/A"
            sigma_val = node.lognorm_sigma
            if sigma_val is not None:
                sigma_str = f"{sigma_val:.2f}" if sigma_val > 1e-3 else f"{sigma_val:.2e}"
            else:
                sigma_str = "N/A"
            print(f"{prefix}, mean={mean_str}, LN(mu={mu_str}, sigma={sigma_str})")
        elif isinstance(node, NodeGPD):
             # 수정된 부분: node.gpd_params가 None이 아닌지 확인
             nll_str = f"{node.nll:.1f}" if node.nll is not None and np.isfinite(node.nll) else "N/A"
             if node.gpd_params is not None and isinstance(node.gpd_params, np.ndarray) and node.gpd_params.shape == (2,):
                 s, g = node.gpd_params
                 # sigma가 0에 가까울 때 과학적 표기법 사용
                 s_str = f"{s:.3f}" if abs(s) > 1e-4 else f"{s:.3e}"
                 g_str = f"{g:.3f}"
                 print(f"{prefix}, GPD(σ={s_str}, γ={g_str}), NLL={nll_str}")
             else:
                 # gpd_params가 None이거나 유효하지 않은 형식일 경우
                 param_status = "None" if node.gpd_params is None else "Invalid Format"
                 print(f"{prefix}, GPD(params={param_status}), NLL={nll_str}")
        else:
            print(f"{prefix}, Unknown Node Type")
    else: # 내부 노드
        # 내부 노드에도 n_samples 추가 (NodeBase에 추가됨)
        n_samples_str = f"N={node.n_samples}" if hasattr(node, 'n_samples') else "N=?"

        # split_var 유효성 검사 추가
        if node.split_var is None or node.split_thr is None:
             print(f"{indent}오류: 내부 노드에 분할 정보가 없습니다. {n_samples_str}")
             return

        feat = feat_names[node.split_var] if feat_names and 0 <= node.split_var < len(feat_names) else f"X{node.split_var}"
        thr = node.split_thr
        gain_info = ""
        # gain 값 형식화 개선 및 None/NaN 처리
        if hasattr(node, 'gain') and node.gain is not None and np.isfinite(node.gain) and node.gain > 0: # L1/L2
             gain_info = f"(Gain={node.gain:.3f})"
        elif hasattr(node, 'split_gain') and node.split_gain is not None and np.isfinite(node.split_gain) and node.split_gain > -float('inf'): # GPD (penalized gain은 음수일 수 있음)
             gain_info = f"(PenalizedGain={node.split_gain:.3f})"

        print(f"{indent}[{feat} ≤ {thr:.4g}] {gain_info} {n_samples_str}")

        # 자식 노드 존재 여부 확인
        if node.left:
            print_tree_structure(node.left, feat_names, indent + "  ")
        else:
             # 자식 노드가 없으면 명시적으로 표시 (오류 상황일 수 있음)
             print(f"{indent}  (Left child missing or pruned)")
        if node.right:
            print_tree_structure(node.right, feat_names, indent + "  ")
        else:
             print(f"{indent}  (Right child missing or pruned)")

# Obsolete print functions replaced by the unified print_tree_structure
# def print_tree_structure_l1(...)
# def print_tree_structure_l2(...)
# def print_tree_structure_gpd(...)


def validate_tree_structure(tree: NodeBase, min_samples_leaf: int = 1) -> bool:
    """Basic validation of tree structure."""
    valid = True
    nodes_to_visit = [tree]
    while nodes_to_visit:
        node = nodes_to_visit.pop(0)
        if node is None: # Should not happen
             logging.error("Validation Error: Encountered None node in tree.")
             return False

        if not hasattr(node, 'is_leaf'):
             logging.error(f"Validation Error: Node missing 'is_leaf' attribute: {node}")
             return False

        if node.is_leaf:
            if not hasattr(node, 'n_samples'):
                logging.warning(f"Validation Warning: Leaf node missing 'n_samples'.")
            elif node.n_samples < min_samples_leaf and node.n_samples > 0: # Allow 0 samples if split failed higher up
                logging.warning(f"Validation Warning: Leaf node has {node.n_samples} samples, less than min {min_samples_leaf}.")
                # valid = False # Optional: make this an error
            if isinstance(node, NodeGPD) and node.gpd_params is None:
                 logging.warning(f"Validation Warning: GPD Leaf node (N={node.n_samples}) has no GPD parameters.")
                 # valid = False # Optional: make this an error

        else: # Internal node
            if node.left is None or node.right is None:
                logging.error(f"Validation Error: Internal node at depth {node.depth} missing children.")
                return False
            if node.split_var is None or node.split_thr is None:
                logging.error(f"Validation Error: Internal node at depth {node.depth} missing split info.")
                return False
            nodes_to_visit.append(node.left)
            nodes_to_visit.append(node.right)

            # Check child samples add up (optional, requires n_samples on internal nodes too)
            # if hasattr(node, 'n_samples') and hasattr(node.left, 'n_samples') and hasattr(node.right, 'n_samples'):
            #     if node.n_samples != node.left.n_samples + node.right.n_samples:
            #          logging.warning(f"Sample count mismatch at node depth {node.depth}: {node.n_samples} != {node.left.n_samples} + {node.right.n_samples}")


    return valid


# ---------------------------------------------------------------------#
# 7. Data Preprocessing Utility
# ---------------------------------------------------------------------#
def create_lagged_features(df: pd.DataFrame,
                           target_col: str,
                           lag_config: Dict[str, List[int]],
                           date_col: str = "Date",
                           start_year: Optional[int] = None,
                           additional_features: Optional[List[str]] = None
                           ) -> Tuple[pd.DataFrame, List[str]]:
    """
    Creates lagged features based on the config dictionary.

    Args:
        df: Input DataFrame with time series data and a date column.
        target_col: Name of the target variable column.
        lag_config: Dictionary {feature_name: [lag1, lag2, ...]}
        date_col: Name of the column containing dates/years.
        start_year: Optional minimum year to keep data from.
        additional_features: List of other non-lagged features to keep.

    Returns:
        Tuple: (DataFrame with target and features, list of feature names used)
    """
    df_proc = df.copy()

    # Ensure date column is usable for sorting/filtering
    if pd.api.types.is_datetime64_any_dtype(df_proc[date_col]):
        df_proc['_year_'] = df_proc[date_col].dt.year
    elif pd.api.types.is_numeric_dtype(df_proc[date_col]): # Assume it's already year if numeric
         df_proc['_year_'] = df_proc[date_col]
    else: # Try converting to datetime then extract year
        try:
            df_proc[date_col] = pd.to_datetime(df_proc[date_col], errors='coerce')
            df_proc['_year_'] = df_proc[date_col].dt.year
            if df_proc['_year_'].isnull().any():
                 raise ValueError("Date conversion resulted in NaNs.")
        except Exception as e:
            raise ValueError(f"Could not interpret date column '{date_col}'. Error: {e}")

    df_proc = df_proc.sort_values(by='_year_')

    lagged_feature_names = []
    all_features_to_keep = [target_col]

    # Create lagged features
    for var, lags in lag_config.items():
        if var not in df_proc.columns:
            logging.warning(f"Column '{var}' for lagging not found in DataFrame. Skipping.")
            continue
        for lag in lags:
            if lag <= 0:
                 logging.warning(f"Skipping non-positive lag {lag} for variable '{var}'.")
                 continue
            lagged_col_name = f"{var}_lag{lag}"
            df_proc[lagged_col_name] = df_proc[var].shift(lag)
            lagged_feature_names.append(lagged_col_name)

    all_features_to_keep.extend(lagged_feature_names)

    # Add other specified features
    if additional_features:
        for add_col in additional_features:
            if add_col not in df_proc.columns:
                logging.warning(f"Additional feature '{add_col}' not found. Skipping.")
            elif add_col not in all_features_to_keep: # Avoid duplicates
                 all_features_to_keep.append(add_col)


    # Filter by start year if specified
    if start_year is not None:
        df_proc = df_proc[df_proc['_year_'] >= start_year].copy()

    # Select final columns and drop rows with NaNs introduced by lagging/filtering
    final_df = df_proc[all_features_to_keep].dropna().copy()

    # Clean up temporary year column if it wasn't an original feature
    if '_year_' in final_df.columns and '_year_' != date_col:
        final_df = final_df.drop(columns=['_year_'])

    # Identify final feature names (excluding the target)
    feature_names_used = [col for col in final_df.columns if col != target_col]

    logging.info(f"Created dataset with {len(final_df)} samples and {len(feature_names_used)} features after lagging and filtering.")

    return final_df, feature_names_used

# create_lagged_df_for_selected_vars -> Renamed to create_lagged_features

# ---------------------------------------------------------------------#
# 8. Main Execution Example
# ---------------------------------------------------------------------#

