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

def merge_small_leaves_gpd(root: NodeGPD,
                           X: np.ndarray,
                           y: np.ndarray,
                           merge_threshold: int = 100) -> NodeGPD:
    """
    Bottom-up 방식으로 작은 리프를 부모 노드로 병합합니다.
      - root: prune_gpd_with_cv 등으로 미리 pruning 된 트리
      - X, y: Tail (초과치) 데이터
      - merge_threshold: 최소 샘플 수 기준
    """
    n = len(y)

    # 1) 모든 샘플을 현재 리프에 할당
    leaves = _get_all_leaves(root)  # 기존 함수 활용
    leaf_of = np.empty(n, dtype=int)
    for i in range(n):
        leaf = find_leaf(root, X[i])  # find_leaf_gpd 대신 범용 find_leaf 사용
        leaf_of[i] = leaves.index(leaf)
    leaf_idx_map = {i: np.where(leaf_of == i)[0] for i in range(len(leaves))}

    # 2) internal nodes (leaf가 아닌 노드만)
    internal_nodes = []
    def collect_internals(node):
        if not node.is_leaf:
            internal_nodes.append(node)
            if node.left:  collect_internals(node.left)
            if node.right: collect_internals(node.right)
    collect_internals(root)

    # 3) bottom-up 병합
    merged = True
    while merged:
        merged = False
        for node in internal_nodes:
            # Left/Right 가 둘 다 존재하고, 둘 다 leaf 여야 병합 가능
            if node.left is None or node.right is None:
                continue
            if not (node.left.is_leaf and node.right.is_leaf):
                continue

            # 각 자식 leaf_id
            left_id  = leaves.index(node.left)
            right_id = leaves.index(node.right)
            idx_l = leaf_idx_map.get(left_id, [])
            idx_r = leaf_idx_map.get(right_id, [])

            # 병합 기준 미만이면 병합
            if min(len(idx_l), len(idx_r)) < merge_threshold:
                combined_idx = np.concatenate([idx_l, idx_r])
                # 파라미터 재적합
                theta = fit_gpd_mle(y[combined_idx])
                nll   = neg_loglik_gpd((theta[0], theta[1]), y[combined_idx])

                # 부모 노드를 leaf로 전환
                node.is_leaf    = True
                node.split_var  = None
                node.split_thr  = None
                node.left       = None
                node.right      = None
                node.gpd_params = theta
                node.nll        = nll

                # leaf 리스트/맵 갱신
                # 기존 child leaf entries 삭제
                leaf_idx_map.pop(left_id, None)
                leaf_idx_map.pop(right_id, None)
                # 새로운 leaf id 추가
                new_id = max(leaf_idx_map.keys(), default=-1) + 1
                leaves.append(node)
                leaf_idx_map[new_id] = combined_idx

                merged = True
                break

    return root


