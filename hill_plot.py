
# thresholding.py ─────────────────────────────────────────────────
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
import numpy as np
import logging
from scipy import stats

MIN_FLOAT = np.finfo(float).eps

@dataclass
class HillDiag:
    k_grid:     np.ndarray  # k 값 벡터
    gamma_hat:  np.ndarray  # smooth된 Hill 추정치
    grad_norm:  np.ndarray  # |∇γ| / |γ|
    var_rel:    np.ndarray  # var(γ) / γ²
    smooth_win: int         # 스무딩 윈도우 크기

def _auto_hill(y: np.ndarray,
               *,
               k_min: int = 50,
               k_max_prop: float = 0.1,
               window: int = 61,
               rel_grad_tol: float = 0.04,
               rel_var_tol: float = 0.03,
               gamma_min: float = 0.02,
               min_plateau_len: int = 10,
               plateau_strategy= 'longest',
               min_excess: Optional[int] = None,
               max_excess: Optional[int] = None
               ) -> Tuple[float, int, float, HillDiag]:
    """
    Hill estimator 기반 자동 임계값 탐색.
    plateau_strategy: 'first' | 'longest'
    min_excess, max_excess: None 이면 len(y)의 1%, 25%로 자동 설정
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if min_excess is None:
        min_excess = max(50, int(n*0.01))
    if max_excess is None:
        max_excess = int(n*0.25)

    # 내림차순 정렬
    y_desc = np.sort(y)[::-1]
    k_max = int(min(n*k_max_prop, n-1))
    ks = np.arange(k_min, k_max+1)
    # 양수만
    y_pos = y_desc[y_desc>0]
    ks = ks[ks < y_pos.size]

    # Hill γ(k)
    gamma_k = np.array([
        np.mean(np.log(y_pos[:k]) - np.log(y_pos[k]))
        for k in ks
    ])

    # 스무딩
    win = window if window<=gamma_k.size else max(3,(gamma_k.size//2)*2+1)
    pad = win//2
    g_pad = np.pad(gamma_k, (pad,pad), mode='edge')
    smooth = np.convolve(g_pad, np.ones(win)/win, mode='valid')

    # 진단량 계산
    grad_norm = np.abs(np.gradient(smooth)) / np.maximum(np.abs(smooth), MIN_FLOAT)
    g2_pad = np.pad(gamma_k**2, (pad,pad), mode='edge')
    mean2 = np.convolve(g2_pad, np.ones(win)/win, mode='valid')
    var_rel = np.maximum(mean2 - smooth**2, 0) / np.maximum(smooth**2, MIN_FLOAT)

    # 안정 구간 boolean mask
    stable = (grad_norm < rel_grad_tol) & (var_rel < rel_var_tol) & (smooth > gamma_min)

    # plateau 블록 탐색
    blocks, cur = [], []
    for i, ok in enumerate(stable):
        if ok:
            cur.append(i)
        else:
            if cur:
                blocks.append(cur.copy())
                cur.clear()
    if cur:
        blocks.append(cur.copy())

    # 블록 선택
    k_hat = None
    valid_blocks = [b for b in blocks if len(b) >= min_plateau_len]
    if valid_blocks:
        if plateau_strategy == 'first':
            blk = valid_blocks[0]
        else:  # 'longest'
            blk = max(valid_blocks, key=len)
        mid = blk[len(blk)//2]
        k_hat = ks[mid]
        logging.info(f"Plateau({plateau_strategy}) -> block indices {blk[:3]}... len={len(blk)} -> k={k_hat}")

    # fallback: KS 통계량 최소화
    if k_hat is None:
        logging.warning("No plateau found: fallback to KS-minimization")
        ks_stats = np.full(ks.shape, np.inf)
        for idx, k in enumerate(ks):
            if k < min_excess or k > max_excess:
                continue
            excess = y_pos[:k] - y_pos[k]
            # 충분한 샘플만
            if excess.size < min_excess:
                continue
            ξ, _, σ = stats.genpareto.fit(excess, floc=0)
            ks_stats[idx] = stats.kstest(excess, 'genpareto', args=(ξ,0,σ))[0]
        best = np.nanargmin(ks_stats)
        k_hat = ks[best]
        logging.info(f"KS-min D={ks_stats[best]:.4f} at k={k_hat}")

    # 경계 보정
    k_hat = int(np.clip(k_hat, min_excess, max_excess))
    k_hat = min(k_hat, y_pos.size-1)

    # 최종 u_hat, γ_hat
    u_hat = float(y_pos[k_hat])
    gamma_hat = np.mean(np.log(y_pos[:k_hat]) - np.log(y_pos[k_hat]))

    diag = HillDiag(
        k_grid=ks,
        gamma_hat=smooth,
        grad_norm=grad_norm,
        var_rel=var_rel,
        smooth_win=win
    )
    return u_hat, k_hat, gamma_hat, diag

def select_threshold(y: np.ndarray,
                     *,
                     u_override: Optional[float] = None,
                     return_diag: bool = False,
                     **hill_kw
                     ) -> Tuple[float,int,float] or Tuple[float,int,float,HillDiag]:
    """
    - u_override: override 기준 임계값을 직접 지정
    - hill_kw: _auto_hill에 넘겨줄 파라미터들
    """
    y = np.asarray(y, dtype=float)
    if y.ndim!=1 or y.size==0:
        raise ValueError("y must be 1-D non-empty array")

    u_auto, k_auto, g_auto, diag = _auto_hill(y, **hill_kw)

    if u_override is not None:
        u_hat = float(u_override)
        exceed = y[y>u_hat]
        k_hat = exceed.size
        if k_hat < 1:
            logging.warning("u_override > max(y): no exceedances")
            gamma_hat = 0.0
        else:
            exc_sorted = np.sort(exceed)[::-1]
            gamma_hat = np.mean(np.log(exc_sorted[:k_hat]) - np.log(exc_sorted[k_hat-1]))
    else:
        u_hat, k_hat, gamma_hat = u_auto, k_auto, g_auto

    if return_diag:
        return u_hat, k_hat, gamma_hat, diag
    return u_hat, k_hat, gamma_hat

