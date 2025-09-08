# thresholding.py ─────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Literal, Union
import numpy as np
import logging
from scipy import stats

# 모듈 전용 로거 (필요시 사용처에서 basicConfig 설정 권장)
logger = logging.getLogger(__name__)

MIN_FLOAT = np.finfo(float).eps


@dataclass
class HillDiag:
    """Hill 추정량 진단을 위한 데이터 클래스"""
    k_grid:     np.ndarray  # k 값 벡터
    gamma_hat:  np.ndarray  # smooth된 Hill 추정치
    grad_norm:  np.ndarray  # |∇γ| / |γ|
    var_rel:    np.ndarray  # var(γ) / γ²
    smooth_win: int         # 스무딩 윈도우 크기


def _find_plateau_blocks(stable: np.ndarray) -> list[list[int]]:
    """
    안정 구간(True 연속 구간)의 인덱스 블록 리스트를 반환.
    예) [F,T,T,F,T] -> [[1,2],[4]]
    """
    stable = np.asarray(stable, dtype=bool)
    if stable.size == 0 or not np.any(stable):
        return []

    padded = np.r_[False, stable, False]
    boundaries = np.diff(padded.astype(int))
    starts = np.where(boundaries == 1)[0]
    ends   = np.where(boundaries == -1)[0]
    return [list(range(s, e)) for s, e in zip(starts, ends)]


def _auto_hill(
    y: np.ndarray,
    *,
    k_min: int = 50,
    k_max_prop: float = 0.1,
    window: int = 61,
    rel_grad_tol: float = 0.04,
    rel_var_tol: float = 0.03,
    gamma_min: float = 0.02,
    min_plateau_len: int = 10,
    plateau_strategy: Literal['first', 'longest'] = 'longest',
    min_excess: Optional[int] = None,
    max_excess: Optional[int] = None
) -> Tuple[float, int, float, HillDiag]:
    """Hill estimator 기반 자동 임계값 탐색."""
    # 1) 입력 정제
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y must be a 1-D non-empty array")

    # 2) 양수 꼬리만 사용
    y_desc = np.sort(y)[::-1]
    y_pos = y_desc[y_desc > 0]

    if y_pos.size < max(3, k_min):
        logger.warning("Not enough positive data points for Hill estimation.")
        diag_empty = HillDiag(np.array([]), np.array([]), np.array([]), np.array([]), 0)
        return np.inf, 0, 0.0, diag_empty

    # 3) k 범위 설정 (y_pos 기준)
    if not (0 < k_max_prop <= 1.0):
        raise ValueError("k_max_prop must be in (0, 1].")
    k_max = int(min(y_pos.size * k_max_prop, y_pos.size - 2))
    if k_max < k_min:
        # 데이터가 작을 때 최소한의 검색 구간 확보
        k_min = max(3, min(k_min, max(3, y_pos.size // 5)))
        k_max = max(k_min + 1, y_pos.size - 2)

    ks = np.arange(k_min, k_max + 1, dtype=int)
    if ks.size == 0:
        logger.warning(f"k_min({k_min}) >= k_max({k_max}). No range to search for k.")
        diag_empty = HillDiag(np.array([]), np.array([]), np.array([]), np.array([]), 0)
        return np.inf, 0, 0.0, diag_empty

    # 4) Hill 곡선: γ_k = mean( log Y_(1..k) - log Y_(k+1) )
    gamma_k = np.array([
        np.mean(np.log(y_pos[:k]) - np.log(y_pos[k]))
        for k in ks
    ])

    # 5) 이동평균 스무딩 (항상 홀수 윈도우, 과도한 크기 방지)
    win = window if (window % 2 == 1) else (window + 1)
    win = min(win, max(3, (gamma_k.size // 2) * 2 + 1))
    pad = win // 2
    g_pad = np.pad(gamma_k, (pad, pad), mode='edge')
    smooth = np.convolve(g_pad, np.ones(win) / win, mode='valid')

    # 6) 진단 지표: 상대 기울기, 상대 분산
    grad_norm = np.abs(np.gradient(smooth)) / np.maximum(np.abs(smooth), MIN_FLOAT)
    g2_pad = np.pad(gamma_k**2, (pad, pad), mode='edge')
    mean2 = np.convolve(g2_pad, np.ones(win) / win, mode='valid')
    var_rel = np.maximum(mean2 - smooth**2, 0.0) / np.maximum(smooth**2, MIN_FLOAT)

    # 7) 안정 구간 및 plateau 탐색
    stable = (grad_norm < rel_grad_tol) & (var_rel < rel_var_tol) & (smooth > gamma_min)
    blocks = _find_plateau_blocks(stable)

    k_hat: Optional[int] = None
    valid_blocks = [b for b in blocks if len(b) >= min_plateau_len]
    if valid_blocks:
        blk = valid_blocks[0] if plateau_strategy == 'first' else max(valid_blocks, key=len)
        mid = blk[len(blk) // 2]
        k_hat = int(ks[mid])
        logger.info(f"Plateau({plateau_strategy}) -> len={len(blk)} -> k={k_hat}")

    # 8) Plateau 실패 시 KS 최소화 (y_pos 기준으로 min/max_excess 설정)
    if k_hat is None:
        logger.warning("No plateau found: fallback to KS-minimization.")
        if min_excess is None:
            min_excess = max(50, int(y_pos.size * 0.01))
        if max_excess is None:
            max_excess = int(y_pos.size * 0.25)
        if min_excess >= max_excess:
            max_excess = max(min_excess + 1, min(y_pos.size - 2, min_excess + 50))

        ks_stats = np.full(ks.shape, np.inf)
        for idx, k in enumerate(ks):
            if k < min_excess or k > max_excess:
                continue

            # 초과분: 0 제거(동일값으로 인한 타이 제거)
            excess = y_pos[:k] - y_pos[k]
            excess = excess[excess > 0]
            if excess.size < min_excess:
                continue

            try:
                shape, loc, scale = stats.genpareto.fit(excess, floc=0.0)
                if scale <= 0:
                    continue
                D = stats.kstest(excess, cdf=stats.genpareto.cdf, args=(shape, loc, scale))[0]
                ks_stats[idx] = D
            except Exception as e:
                logger.debug(f"GPD fit failed for k={k}: {e}")
                continue

        if np.all(np.isinf(ks_stats)):
            logger.error("KS-minimization also failed; falling back to min_excess.")
            k_hat = int(min_excess)
        else:
            best = int(np.nanargmin(ks_stats))
            k_hat = int(ks[best])
            logger.info(f"KS-min: D={ks_stats[best]:.4f} at k={k_hat}")

    # 9) 최종 안전 클리핑
    if min_excess is not None:
        k_hat = max(k_hat, int(min_excess))
    if max_excess is not None:
        k_hat = min(k_hat, int(max_excess))
    k_hat = min(k_hat, y_pos.size - 2)

    # 10) 임계값, 최종 gamma
    u_hat = float(y_pos[k_hat])
    gamma_hat = float(np.mean(np.log(y_pos[:k_hat]) - np.log(y_pos[k_hat])))

    diag = HillDiag(k_grid=ks, gamma_hat=smooth, grad_norm=grad_norm, var_rel=var_rel, smooth_win=win)
    return u_hat, k_hat, gamma_hat, diag


def select_threshold(
    y: np.ndarray,
    *,
    u_override: Optional[float] = None,
    return_diag: bool = False,
    **hill_kw
) -> Union[Tuple[float, int, float], Tuple[float, int, float, HillDiag]]:
    """
    데이터에 대한 임계값을 선택합니다.
    - u_override: 임계값을 수동으로 지정.
    - hill_kw: _auto_hill에 전달할 파라미터.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y must be a 1-D non-empty array")

    u_auto, k_auto, g_auto, diag = _auto_hill(y, **hill_kw)

    if u_override is not None:
        if not np.isfinite(u_override) or u_override <= 0:
            raise ValueError("u_override must be a finite positive number.")
        u_hat = float(u_override)
        exceed_vals = y[y > u_hat]
        k_hat = int(exceed_vals.size)

        if k_hat < 2:
            logger.warning(
                f"u_override ({u_override}) -> exceedances={k_hat}. "
                "Gamma estimate may be unreliable; set to 0.0."
            )
            gamma_hat = 0.0
        else:
            gamma_hat = float(np.mean(np.log(exceed_vals) - np.log(u_hat)))
    else:
        u_hat, k_hat, gamma_hat = u_auto, k_auto, g_auto

    if return_diag:
        return u_hat, k_hat, gamma_hat, diag
    return u_hat, k_hat, gamma_hat


__all__ = ["HillDiag", "select_threshold", "_auto_hill", "_find_plateau_blocks"]
