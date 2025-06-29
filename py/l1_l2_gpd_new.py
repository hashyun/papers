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
FLT_INFO = np.finfo(float)

SEED        = 42
rng         = np.random.default_rng(SEED)
MIN_FLOAT   = FLT_INFO.eps
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
        self.n_samples: int = 0

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
        self.gpd_params: Optional[Tuple[float, float]] = None # (sigma, gamma)
        self.nll: Optional[float] = None # Negative Log-Likelihood at this node

# ---------------------------------------------------------------------#
# 1. Distribution Fitting Functions (Lognormal, GPD)
# ---------------------------------------------------------------------#

def _nll_lognormal(params: Tuple[float, float],
                   y: np.ndarray,
                   *, trunc_left: float = 0.0) -> float:
    """Negative Log-Likelihood for (optionally left-truncated) LogNormal."""
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
        zc   = (np.log(trunc_left) - mu) / sigma
        surv = 1.0 - stats.norm.cdf(zc)
        if surv <= MIN_FLOAT:
            return LARGE_FLOAT
        base -= y_pos.size * np.log(surv)

    return base if np.isfinite(base) else LARGE_FLOAT

def fit_lognormal_mle(y: np.ndarray, *, trunc_left: float = 0.0
                      ) -> Tuple[float, float]:
    """MLE for (optionally left-truncated) LogNormal -> (μ̂, σ̂)."""
    y_pos = y[y > 0]
    n_pos = y_pos.size
    if n_pos < 2:
        return (np.log(y_pos[0]), MIN_FLOAT*10) if n_pos == 1 else (0.0, 1.0)

    log_y = np.log(y_pos)
    init   = [log_y.mean(), max(log_y.std(ddof=1), MIN_FLOAT*10)]
    bounds = [(None, None), (MIN_FLOAT*10, None)]

    obj = lambda p: _nll_lognormal(p, y_pos, trunc_left=trunc_left)

    try:
        res = minimize(obj, init, method="L-BFGS-B", bounds=bounds)
        if res.success and np.all(np.isfinite(res.x)):
            μ, σ = res.x
            return float(μ), float(max(σ, MIN_FLOAT*100))
    except Exception as e:
        logging.debug(f"LogNormal MLE failed: {e}")

    return tuple(init)

def neg_loglik_gpd(theta: Tuple[float, float], y: np.ndarray) -> float:
    """Calculate negative log-likelihood for Generalized Pareto Distribution (GPD)."""
    sigma, gamma = theta
    n = len(y)
    if n == 0:
        return 0.0
    if sigma <= MIN_FLOAT:
        return LARGE_FLOAT

    if abs(gamma) < 1e-6: # Exponential distribution case
        if np.any(y < 0):
             return LARGE_FLOAT
        nll = n * np.log(sigma) + np.sum(y) / sigma
    else: # Standard GPD case
        z = 1 + gamma * y / sigma
        if np.any(z <= MIN_FLOAT):
            return LARGE_FLOAT
        nll = n * np.log(sigma) + (1 / gamma + 1) * np.sum(np.log(z))

    return nll if np.isfinite(nll) else LARGE_FLOAT

def fit_gpd_mle(y: np.ndarray,
                *, gamma_bounds: Tuple[float, float] = (-0.5, 0.95)) -> np.ndarray:
    """Stabilized GPD MLE fitting."""
    n = len(y)
    if n < 30:
        fallback_sigma = max(np.median(y), MIN_FLOAT * 10) if n > 0 else 1.0
        return np.array([fallback_sigma, -0.2])

    # PWM initial estimate
    y_sorted = np.sort(y)
    y_bar = np.mean(y_sorted)
    pwm1 = np.mean((1 - (np.arange(n) + 0.65) / n) * y_sorted) # Unbiased PWM est.
    
    gamma0_raw = 2 - y_bar / (y_bar - 2 * pwm1 + MIN_FLOAT)
    gamma0 = np.clip(gamma0_raw, -0.4, 0.9)
    sigma0 = max((2 * y_bar * pwm1) / (y_bar - 2 * pwm1 + MIN_FLOAT), MIN_FLOAT * 10)
    
    init_params = np.array([sigma0, gamma0])
    bounds = [(MIN_FLOAT*10, None), gamma_bounds]

    def objective(theta: np.ndarray, data: np.ndarray) -> float:
        return neg_loglik_gpd(theta, data)

    try:
        res = minimize(
            objective, x0=init_params, args=(y,), method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-9, "eps": 1e-7},
        )
        if res.success and np.isfinite(res.fun):
            s_opt, g_opt = res.x
            return np.array([max(s_opt, MIN_FLOAT * 10), g_opt])
        else:
             logging.debug(f"GPD MLE optimization failed. Returning PWM init.")
             return init_params
    except Exception as e:
        logging.debug(f"GPD MLE failed with error: {e}. Returning PWM init.")
        return init_params

def node_cost_gpd(y: np.ndarray) -> Tuple[float, np.ndarray]:
    """Calculate node cost (NLL) and parameters for GPD."""
    if len(y) == 0:
        return 0.0, np.array([1.0, 0.0])

    try:
        theta = fit_gpd_mle(y)
        cost = neg_loglik_gpd(theta, y)
        if not np.isfinite(cost):
            return LARGE_FLOAT, theta
        return cost, theta
    except Exception as e:
        logging.warning(f"Error during node_cost_gpd: {e}. Returning large cost.")
        fallback_sigma = max(np.median(y), MIN_FLOAT*10) if len(y) > 0 else 1.0
        return LARGE_FLOAT, np.array([fallback_sigma, -0.2])

# ---------------------------------------------------------------------#
# 2. CART Splitting Functions
# ---------------------------------------------------------------------#
def _best_split_l1(x: np.ndarray, y: np.ndarray, min_leaf: int = 1
                   ) -> Tuple[float, Optional[float]]:
    """Find the best split point for L1 CART (median-based)."""
    n = len(y)
    if n < 2 * min_leaf:
        return 0.0, None

    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]

    parent_loss = mae_loss(ys, np.median(ys))
    best_gain, best_thr = 0.0, None

    for i in range(min_leaf, n - min_leaf):
        if xs[i] == xs[i - 1]:
            continue
        
        left_y, right_y = ys[:i], ys[i:]
        child_loss = mae_loss(left_y, np.median(left_y)) + mae_loss(right_y, np.median(right_y))
        gain = parent_loss - child_loss

        if gain > best_gain:
            best_gain = gain
            best_thr = 0.5 * (xs[i] + xs[i - 1])

    return (best_gain, best_thr) if best_gain > MIN_FLOAT else (0.0, None)

def _best_split_l2(x: np.ndarray, y: np.ndarray, min_leaf: int = 1
                   ) -> Tuple[float, Optional[float]]:
    """Find the best split point for L2 CART (mean-based) efficiently."""
    n = len(y)
    if n < 2 * min_leaf:
        return 0.0, None

    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]

    parent_loss = np.sum((ys - np.mean(ys)) ** 2)
    
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
        
        gain = parent_loss - (loss_l + loss_r)

        if gain > best_gain:
            best_gain = gain
            best_thr = 0.5 * (xs[i] + xs[i - 1])

    return (best_gain, best_thr) if best_gain > MIN_FLOAT else (0.0, None)

def _best_split_gpd(x: np.ndarray, y: np.ndarray, min_leaf: int = 30,
                    penalty: float = 2.0) -> Tuple[float, Optional[float]]:
    """
    Finds the best split for GPD-CART using NLL reduction.
    Uses AIC-like penalty for complexity.
    """
    n = len(y)
    if n < 2 * min_leaf:
        return 0.0, None

    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]

    parent_nll, _ = node_cost_gpd(ys)
    if not np.isfinite(parent_nll):
         logging.warning("Parent GPD NLL is not finite. Cannot split.")
         return 0.0, None

    best_gain, best_thr = -LARGE_FLOAT, None

    for i in range(min_leaf, n - min_leaf):
        if xs[i] == xs[i - 1]:
            continue

        left_y, right_y = ys[:i], ys[i:]
        l_nll, _ = node_cost_gpd(left_y)
        r_nll, _ = node_cost_gpd(right_y)

        if not (np.isfinite(l_nll) and np.isfinite(r_nll)):
            continue
        
        # Penalized Gain = ParentNLL - ChildNLL_Sum - Penalty
        # Penalty is for adding a split (2 new child nodes).
        # AIC-style penalty = 2 * k, where k = #params. Here k = 2 params per child.
        # So total penalty = penalty * 2.
        current_penalty = penalty * 2 
        gain = parent_nll - (l_nll + r_nll) - current_penalty

        if gain > best_gain:
            best_gain = gain
            best_thr = 0.5 * (xs[i] + xs[i - 1])

    if best_gain <= 0:
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

    if n_samples < 2 * min_leaf or depth >= max_depth or n_samples == 0:
        return node

    best_gain, best_var, best_thr = -1.0, None, None

    for j in range(n_features):
        gain, thr = _best_split_l1(X[:, j], y, min_leaf)
        if thr is not None and gain > best_gain:
            best_gain, best_var, best_thr = gain, j, thr

    if best_thr is None or best_gain <= MIN_FLOAT:
        return node

    mask_left = X[:, best_var] <= best_thr
    n_left, n_right = np.sum(mask_left), n_samples - np.sum(mask_left)

    if n_left < min_leaf or n_right < min_leaf:
        return node

    node.is_leaf = False
    node.split_var, node.split_thr, node.gain = best_var, best_thr, best_gain
    
    logging.debug(f"Depth {depth}: Splitting '{feature_names[best_var]}' <= {best_thr:.3g}, Gain={best_gain:.3f}, N={n_samples}->({n_left}, {n_right})")
    node.left = grow_tree_l1(X[mask_left], y[mask_left], feature_names, min_leaf, max_depth, depth + 1)
    node.right = grow_tree_l1(X[~mask_left], y[~mask_left], feature_names, min_leaf, max_depth, depth + 1)
    return node

def grow_tree_l2(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                 min_leaf: int = 30, max_depth: int = 4, depth: int = 0) -> NodeL2:
    """Grows an L2 CART (mean-based) recursively."""
    n_samples, n_features = X.shape
    node = NodeL2()
    node.depth, node.n_samples = depth, n_samples
    node.mean_val = np.mean(y) if n_samples > 0 else 0.0

    if n_samples < 2 * min_leaf or depth >= max_depth or n_samples == 0:
        return node

    best_gain, best_var, best_thr = -1.0, None, None

    for j in range(n_features):
        gain, thr = _best_split_l2(X[:, j], y, min_leaf)
        if thr is not None and gain > best_gain:
            best_gain, best_var, best_thr = gain, j, thr

    if best_thr is None or best_gain <= MIN_FLOAT:
        return node

    mask_left = X[:, best_var] <= best_thr
    n_left, n_right = np.sum(mask_left), n_samples - np.sum(mask_left)

    if n_left < min_leaf or n_right < min_leaf:
        return node

    node.is_leaf = False
    node.split_var, node.split_thr, node.gain = best_var, best_thr, best_gain

    logging.debug(f"Depth {depth}: Splitting '{feature_names[best_var]}' <= {best_thr:.3g}, Gain={best_gain:.3f}, N={n_samples}->({n_left}, {n_right})")
    node.left = grow_tree_l2(X[mask_left], y[mask_left], feature_names, min_leaf, max_depth, depth + 1)
    node.right = grow_tree_l2(X[~mask_left], y[~mask_left], feature_names, min_leaf, max_depth, depth + 1)
    return node

def grow_tree_gpd(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                  min_leaf: int = 100, max_depth: int = 3, depth: int = 0) -> NodeGPD:
    """Grows a GPD CART recursively."""
    n_samples, n_features = X.shape
    node = NodeGPD()
    node.depth, node.n_samples = depth, n_samples
    node.nll, node.gpd_params = node_cost_gpd(y)

    if n_samples < 2 * min_leaf or depth >= max_depth or n_samples == 0:
         logging.debug(f"Depth {depth}: Leaf (Stop condition). N={n_samples}, NLL={node.nll:.2f}")
         return node

    if not np.isfinite(node.nll):
         logging.warning(f"Depth {depth}: Leaf (GPD fit failed). N={n_samples}")
         return node

    best_gain, best_var, best_thr = -LARGE_FLOAT, None, None

    for j in range(n_features):
        gain, thr = _best_split_gpd(X[:, j], y, min_leaf)
        if thr is not None and gain > best_gain:
            best_gain, best_var, best_thr = gain, j, thr

    if best_thr is None or best_gain <= 0:
        logging.debug(f"Depth {depth}: Leaf (No positive gain split). N={n_samples}, NLL={node.nll:.2f}")
        return node

    mask_left = X[:, best_var] <= best_thr
    n_left, n_right = np.sum(mask_left), n_samples - np.sum(mask_left)

    if n_left < min_leaf or n_right < min_leaf:
        logging.debug(f"Depth {depth}: Leaf (Split violates min_leaf). N={n_samples}, NLL={node.nll:.2f}")
        return node

    node.is_leaf = False
    node.split_var, node.split_thr, node.split_gain = best_var, best_thr, best_gain

    logging.debug(f"Depth {depth}: Splitting '{feature_names[best_var]}' <= {best_thr:.3g}, Penalized Gain={best_gain:.3f}, N={n_samples}->({n_left}, {n_right})")
    node.left = grow_tree_gpd(X[mask_left], y[mask_left], feature_names, min_leaf, max_depth, depth + 1)
    node.right = grow_tree_gpd(X[~mask_left], y[~mask_left], feature_names, min_leaf, max_depth, depth + 1)
    return node

# ---------------------------------------------------------------------#
# 4. Leaf Finding and Parameter Assignment
# ---------------------------------------------------------------------#
def find_leaf(node: Union[NodeL1, NodeL2, NodeGPD], x: np.ndarray) -> Union[NodeL1, NodeL2, NodeGPD]:
    """Finds the leaf node for a single data point."""
    current_node = node
    while not current_node.is_leaf:
        if x[current_node.split_var] <= current_node.split_thr:
            current_node = current_node.left
        else:
            current_node = current_node.right
    return current_node

def _get_all_leaves(node: Union[NodeL1, NodeL2, NodeGPD]) -> List[Union[NodeL1, NodeL2, NodeGPD]]:
    """Recursively collect all leaf nodes."""
    leaves = []
    if node.is_leaf:
        leaves.append(node)
    else:
        if node.left: leaves.extend(_get_all_leaves(node.left))
        if node.right: leaves.extend(_get_all_leaves(node.right))
    return leaves

def assign_lognorm_params(root: Union[NodeL1, NodeL2],
                          X: np.ndarray, y: np.ndarray,
                          *, trunc_left: float = 0.0) -> None:
    """Fit LN(μ,σ) per leaf; supports left truncation."""
    if root is None: return

    leaves = _get_all_leaves(root)
    leaf_map = {id(l): i for i, l in enumerate(leaves)}
    leaf_indices = np.array([leaf_map[id(find_leaf(root, x))] for x in X])

    for i, lf in enumerate(leaves):
        y_leaf = y[leaf_indices == i]
        lf.n_samples = y_leaf.size
        if y_leaf.size > 1:
            lf.lognorm_mu, lf.lognorm_sigma = fit_lognormal_mle(y_leaf, trunc_left=trunc_left)
        else:
            lf.lognorm_mu, lf.lognorm_sigma = 0.0, 1.0
            if isinstance(lf, NodeL1): lf.median_val = np.median(y_leaf) if y_leaf.size > 0 else 0.0
            if isinstance(lf, NodeL2): lf.mean_val   = np.mean(y_leaf) if y_leaf.size > 0 else 0.0

# ---------------------------------------------------------------------#
# 5. GPD Tree Pruning (Cost-Complexity Pruning with CV)
# ---------------------------------------------------------------------#

def get_subtree_nll(tree: NodeGPD, X: np.ndarray, y: np.ndarray) -> float:
    """Calculates the total NLL of data (X, y) on a GPD subtree."""
    total_nll = 0.0
    if len(X) == 0: return 0.0

    for i in range(len(X)):
        leaf = find_leaf(tree, X[i])
        if leaf.gpd_params is None:
            total_nll += LARGE_FLOAT
            continue
        
        point_nll = neg_loglik_gpd(leaf.gpd_params, np.array([y[i]]))
        total_nll += point_nll if np.isfinite(point_nll) else LARGE_FLOAT
        
    return total_nll

def _get_pruning_sequence(node: NodeGPD) -> List[Tuple[float, NodeGPD]]:
    """Generates (alpha, node) tuples for cost-complexity pruning."""
    sequence = []
    
    def get_leaves_and_cost(n: NodeGPD):
        if n.is_leaf:
            return 1, n.nll
        
        left_leaves, left_nll = get_leaves_and_cost(n.left)
        right_leaves, right_nll = get_leaves_and_cost(n.right)
        
        total_leaves = left_leaves + right_leaves
        total_leaf_nll = left_nll + right_nll
        
        if total_leaves > 1 and np.isfinite(n.nll) and np.isfinite(total_leaf_nll):
            # alpha = (Cost_of_Node - Cost_of_Subtree) / (Num_Leaves_in_Subtree - 1)
            alpha = (n.nll - total_leaf_nll) / (total_leaves - 1)
            if alpha >= -MIN_FLOAT:
                 sequence.append((alpha, n))

        return total_leaves, total_leaf_nll

    if not node.is_leaf:
        get_leaves_and_cost(node)
        sequence.sort(key=lambda item: item[0])
        
    return sequence

def prune_gpd_single_alpha(tree_root: NodeGPD, alpha: float) -> NodeGPD:
    """
    Prunes a tree for a single alpha. This function prunes nodes
    where the cost-complexity measure favors turning the node into a leaf.
    It returns a pruned COPY of the tree.
    """
    pruned_tree = copy.deepcopy(tree_root)
    
    nodes_to_check = [pruned_tree]
    
    while nodes_to_check:
        node = nodes_to_check.pop(0)
        
        if not node.is_leaf:
            # Recursively add children to the check list
            nodes_to_check.append(node.left)
            nodes_to_check.append(node.right)

            # Sum of NLLs of terminal nodes in the subtree rooted at this node
            terminal_leaves = _get_all_leaves(node)
            cost_of_subtree = sum(leaf.nll for leaf in terminal_leaves if leaf.nll is not None and np.isfinite(leaf.nll))
            num_leaves = len(terminal_leaves)

            # Cost-complexity of keeping the node as a leaf
            cost_as_leaf = node.nll
            
            # Cost-complexity of keeping the subtree
            cost_as_subtree = cost_of_subtree + alpha * num_leaves
            
            # Prune if the leaf version is cheaper
            if cost_as_leaf <= cost_as_subtree:
                node.is_leaf = True
                node.left = None
                node.right = None
                node.split_var = None
                node.split_thr = None
                node.split_gain = 0.0

    return pruned_tree

def prune_gpd_with_cv(root: NodeGPD, X: np.ndarray, y: np.ndarray,
                      n_folds: int = 5, random_state: int = SEED
                      ) -> Tuple[NodeGPD, float]:
    """
    Prunes a GPD tree using cost-complexity with cross-validation.
    """
    n_samples = len(y)
    if n_samples < n_folds * 10 or n_samples < 50:
        logging.warning("Dataset too small for reliable CV pruning. Returning unpruned tree.")
        return root, 0.0

    # 1. Generate sequence of alphas from the full tree
    pruning_seq = _get_pruning_sequence(root)
    alphas = sorted(list(set([alpha for alpha, node in pruning_seq if alpha < LARGE_FLOAT])))
    candidate_alphas = [0.0] + alphas
    if alphas:
         candidate_alphas.append(alphas[-1] * 1.1 + MIN_FLOAT)
    logging.info(f"Generated {len(candidate_alphas)} candidate alphas for CV pruning.")

    # 2. Cross-Validation
    try:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        folds = list(kf.split(X, (y > np.median(y)).astype(int)))
    except ValueError:
         kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
         folds = list(kf.split(X))

    # 3. Evaluate each alpha using CV
    cv_scores = {alpha: [] for alpha in candidate_alphas}

    for fold_idx, (train_indices, test_indices) in enumerate(folds):
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        # Grow a new tree on the training fold
        train_tree = grow_tree_gpd(X_train, y_train, feature_names=[], 
                                   min_leaf=max(int(root.n_samples * 0.05), 30), 
                                   max_depth=root.depth)

        if len(_get_all_leaves(train_tree)) <=1 : continue # Skip if tree is trivial

        for alpha in candidate_alphas:
            pruned_tree = prune_gpd_single_alpha(train_tree, alpha)
            test_nll = get_subtree_nll(pruned_tree, X_test, y_test)
            
            if np.isfinite(test_nll) and len(y_test) > 0:
                 cv_scores[alpha].append(test_nll / len(y_test))
            else:
                 cv_scores[alpha].append(LARGE_FLOAT)

    # 4. Select the best alpha
    mean_cv_scores = {alpha: np.mean(scores) for alpha, scores in cv_scores.items() if scores}
    if not mean_cv_scores:
         logging.error("CV pruning failed. Returning unpruned tree.")
         return root, 0.0

    best_alpha = min(mean_cv_scores, key=mean_cv_scores.get)
    logging.info(f"CV Results: Best alpha = {best_alpha:.4g} with mean NLL = {mean_cv_scores[best_alpha]:.4f}")

    # 5. Prune the original tree with the best alpha
    final_pruned_tree = prune_gpd_single_alpha(root, best_alpha)
    return final_pruned_tree, best_alpha

# ---------------------------------------------------------------------#
# 6. Tree Visualization and Evaluation
# ---------------------------------------------------------------------#
def print_tree_structure(node: Union[NodeL1, NodeL2, NodeGPD],
                         feat_names: List[str], indent: str = ""):
    """Prints the structure of any supported tree type."""
    if not hasattr(node, 'is_leaf'):
         print(f"{indent}Error: Invalid node object.")
         return

    n_samples_str = f"N={node.n_samples}" if hasattr(node, 'n_samples') else "N=?"

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
             if node.gpd_params is not None and isinstance(node.gpd_params, (list, np.ndarray)) and len(node.gpd_params) == 2:
                 s, g = node.gpd_params
                 s_str = f"{s:.3f}" if abs(s) > 1e-4 else f"{s:.3e}"
                 print(f"{prefix}, GPD(σ={s_str}, γ={g:.3f}), NLL={nll_str}")
             else:
                 print(f"{prefix}, GPD(params=None), NLL={nll_str}")
        else:
            print(f"{prefix}, Unknown Node Type")
    else: # Internal node
        if node.split_var is None or node.split_thr is None:
             print(f"{indent}Error: Internal node missing split info. {n_samples_str}")
             return

        feat = feat_names[node.split_var] if feat_names and 0 <= node.split_var < len(feat_names) else f"X{node.split_var}"
        gain_info = ""
        if hasattr(node, 'gain') and node.gain > 0: # L1/L2
             gain_info = f"(Gain={node.gain:.3f})"
        elif hasattr(node, 'split_gain'): # GPD
             gain_info = f"(PenalizedGain={node.split_gain:.3f})"

        print(f"{indent}[{feat} ≤ {node.split_thr:.4g}] {gain_info} {n_samples_str}")

        if node.left:
            print_tree_structure(node.left, feat_names, indent + "  ")
        else:
             print(f"{indent}  (Left child missing or pruned)")
        if node.right:
            print_tree_structure(node.right, feat_names, indent + "  ")
        else:
             print(f"{indent}  (Right child missing or pruned)")

def validate_tree_structure(tree: NodeBase, min_samples_leaf: int = 1) -> bool:
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

# ---------------------------------------------------------------------#
# 7. Data Preprocessing Utility (FUNCTIONALITY MERGED)
# ---------------------------------------------------------------------#
def create_lagged_features(df: pd.DataFrame,
                           target_col: str,
                           lag_config: Dict[str, List[int]],
                           date_col: str,
                           start_year: Optional[int] = None,
                           additional_features: Optional[List[str]] = None
                           ) -> Tuple[pd.DataFrame, List[str]]:
    """
    Creates lagged and log-differenced lagged features.
    
    Args:
        lag_config: Dict like {'var': [1,2], 'var2_diff': [1]}. 
                    '_diff' suffix triggers log-differencing.
    """
    df_proc = df.copy()

    # Handle date column for filtering and sorting
    if not pd.api.types.is_datetime64_any_dtype(df_proc[date_col]):
        df_proc[date_col] = pd.to_datetime(df_proc[date_col], errors='coerce')
    
    df_proc = df_proc.sort_values(by=date_col).reset_index(drop=True)
    
    # Create features
    lagged_feature_names = []
    for var, lags in lag_config.items():
        if var.endswith("_diff"):
            base_var = var.removesuffix("_diff")
            if base_var not in df_proc.columns:
                logging.warning(f"Column '{base_var}' for differencing not found. Skipping.")
                continue
            
            # Calculate log-difference
            log_series = np.log(df_proc[base_var].replace(0, MIN_FLOAT))
            diff_series = log_series.diff()

            for lag in lags:
                if lag <= 0: continue
                lagged_col_name = f"{var}_lag{lag}"
                df_proc[lagged_col_name] = diff_series.shift(lag)
                lagged_feature_names.append(lagged_col_name)

        else: # Standard lagging
            if var not in df_proc.columns:
                logging.warning(f"Column '{var}' for lagging not found. Skipping.")
                continue
            for lag in lags:
                if lag <= 0: continue
                lagged_col_name = f"{var}_lag{lag}"
                df_proc[lagged_col_name] = df_proc[var].shift(lag)
                lagged_feature_names.append(lagged_col_name)

    # Combine all desired columns
    all_feature_names = lagged_feature_names + (additional_features or [])
    final_cols = [target_col] + list(dict.fromkeys(all_feature_names)) # Keep unique feature names
    
    # Filter by start year and drop NaNs
    if start_year is not None:
        df_proc = df_proc[df_proc[date_col].dt.year >= start_year].copy()

    final_df = df_proc[final_cols].dropna().reset_index(drop=True)

    # Final feature names (excluding target)
    feature_names_used = [col for col in final_df.columns if col != target_col]

    logging.info(f"Created dataset with {len(final_df)} samples and {len(feature_names_used)} features.")
    
    return final_df, feature_names_used

# ---------------------------------------------------------------------#
# 8. Main Execution Example
# ---------------------------------------------------------------------#

if __name__ == '__main__':
    # This is an example of how to use the functions.
    # It requires a sample DataFrame `df`.
    
    # 1. Create Sample Data
    n_samples = 1000
    dates = pd.to_datetime(pd.date_range(start='2000-01-01', periods=n_samples, freq='Y'))
    
    # Create some features
    X1 = np.linspace(0, 10, n_samples)
    X2 = np.sin(X1) + rng.normal(0, 0.2, n_samples)
    cat_feat = rng.choice(['A', 'B', 'C'], size=n_samples)
    
    # Create a target variable 'Y'
    # Let's make it so Y is larger when X1 > 5 and cat_feat is 'C'
    y_base = 10 * np.exp(0.2 * X1 + (cat_feat == 'C') * 1.5)
    # Add some heavy-tailed noise (using a Pareto distribution)
    y_noise = (rng.pareto(a=2.5, size=n_samples) + 1) * 20
    Y = y_base + y_noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Y_target': Y,
        'Feature1': X1,
        'Feature2': X2,
        'Category': cat_feat
    })

    # 2. Preprocess Data using the updated function
    lag_config = {
        'Feature1': [1, 2],
        'Feature2_diff': [1] # Lagged log-difference of Feature2
    }
    
    model_df, feature_names = create_lagged_features(
        df=df,
        target_col='Y_target',
        lag_config=lag_config,
        date_col='Date',
        start_year=2003,
        additional_features=['Category'] # Note: Categorical features need encoding
    )
    
    # One-hot encode the categorical feature
    model_df = pd.get_dummies(model_df, columns=['Category'], drop_first=True)
    feature_names = [c for c in model_df.columns if c != 'Y_target']

    X = model_df[feature_names].values
    y = model_df['Y_target'].values

    # 3. Grow and Analyze a GPD Tree
    logging.info("Growing GPD Tree...")
    # For GPD, we need to model the exceedances over a threshold
    threshold = np.quantile(y, 0.85)
    
    exceedances_mask = y > threshold
    X_exceed = X[exceedances_mask]
    y_exceed = y[exceedances_mask] - threshold # Use excesses for fitting

    # Grow the tree on exceedances
    gpd_tree = grow_tree_gpd(X_exceed, y_exceed, feature_names, min_leaf=30, max_depth=3)

    print("\n--- Initial GPD Tree Structure ---")
    print_tree_structure(gpd_tree, feature_names)

    # 4. Prune the GPD Tree with CV
    logging.info("Pruning GPD Tree with Cross-Validation...")
    pruned_gpd_tree, best_alpha = prune_gpd_with_cv(gpd_tree, X_exceed, y_exceed, n_folds=3)
    
    print(f"\n--- Pruned GPD Tree Structure (Best Alpha: {best_alpha:.4g}) ---")
    print_tree_structure(pruned_gpd_tree, feature_names)

    # 5. Grow an L2 (Mean-based) Tree on the full data
    logging.info("\nGrowing L2 (Mean) Tree...")
    l2_tree = grow_tree_l2(X, y, feature_names, min_leaf=50, max_depth=4)

    print("\n--- L2 Tree Structure ---")
    print_tree_structure(l2_tree, feature_names)
