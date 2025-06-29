# thresholding.py ─────────────────────────────────────────────────
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
import numpy as np
import logging
from scipy import stats

MIN_FLOAT = np.finfo(float).eps

# CHANGELOG: 로깅 설정 추가 (테스트 및 정보 확인에 용이)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    CHANGELOG: NumPy를 사용하여 안정 구간(True가 연속되는) 블록을 찾는 효율적인 함수
    """
    if not np.any(stable):
        return []
    
    # True/False가 바뀌는 지점을 찾음
    boundaries = np.diff(stable.astype(int))
    starts = np.where(boundaries == 1)[0] + 1
    ends = np.where(boundaries == -1)[0]
    
    # 블록 리스트 생성
    blocks = []
    
    # 첫 번째 블록 처리 (0에서 시작하는 경우)
    if stable[0]:
        first_end = ends[0] if ends.size > 0 else len(stable) -1
        blocks.append(list(range(0, first_end + 1)))

    # 중간 블록들 처리
    for i, start_idx in enumerate(starts):
        # start_idx 다음에 끝이 있는지 확인
        valid_ends = ends[ends > start_idx]
        if valid_ends.size > 0:
            end_idx = valid_ends[0]
            blocks.append(list(range(start_idx, end_idx + 1)))
            
    # 마지막 블록 처리 (True로 끝나는 경우)
    if stable[-1] and starts.size > 0 and (ends.size == 0 or starts[-1] > ends[-1]):
         blocks.append(list(range(starts[-1], len(stable))))
         
    return blocks

def _auto_hill(y: np.ndarray,
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
    y = np.asarray(y, dtype=float)
    n = y.size
    
    if min_excess is None:
        min_excess = max(50, int(n * 0.01))
    if max_excess is None:
        max_excess = int(n * 0.25)

    y_desc = np.sort(y)[::-1]
    y_pos = y_desc[y_desc > 0]
    
    # CHANGELOG: 데이터 부족 또는 양수 데이터 없는 경우에 대한 안정성 강화
    if y_pos.size < k_min:
        logging.warning("Not enough positive data points for Hill estimation.")
        diag_empty = HillDiag(np.array([]), np.array([]), np.array([]), np.array([]), 0)
        return np.inf, 0, 0.0, diag_empty

    k_max = int(min(n * k_max_prop, n - 1, y_pos.size - 2))
    ks = np.arange(k_min, k_max + 1)
    
    if ks.size == 0:
        logging.warning(f"k_min({k_min}) >= k_max({k_max}). No range to search for k.")
        diag_empty = HillDiag(np.array([]), np.array([]), np.array([]), np.array([]), 0)
        return np.inf, 0, 0.0, diag_empty

    gamma_k = np.array([
        np.mean(np.log(y_pos[:k]) - np.log(y_pos[k]))
        for k in ks
    ])

    win = window if window <= gamma_k.size else max(3, (gamma_k.size // 2) * 2 + 1)
    pad = win // 2
    g_pad = np.pad(gamma_k, (pad, pad), mode='edge')
    smooth = np.convolve(g_pad, np.ones(win) / win, mode='valid')

    grad_norm = np.abs(np.gradient(smooth)) / np.maximum(np.abs(smooth), MIN_FLOAT)
    g2_pad = np.pad(gamma_k**2, (pad, pad), mode='edge')
    mean2 = np.convolve(g2_pad, np.ones(win) / win, mode='valid')
    var_rel = np.maximum(mean2 - smooth**2, 0) / np.maximum(smooth**2, MIN_FLOAT)

    stable = (grad_norm < rel_grad_tol) & (var_rel < rel_var_tol) & (smooth > gamma_min)
    
    # CHANGELOG: 효율적인 방식으로 plateau 블록 탐색
    blocks = _find_plateau_blocks(stable)

    k_hat = None
    valid_blocks = [b for b in blocks if len(b) >= min_plateau_len]
    if valid_blocks:
        if plateau_strategy == 'first':
            blk = valid_blocks[0]
        else:  # 'longest'
            blk = max(valid_blocks, key=len)
        mid = blk[len(blk) // 2]
        k_hat = ks[mid]
        logging.info(f"Plateau({plateau_strategy}) -> block indices {blk[:3]}... len={len(blk)} -> k={k_hat}")

    if k_hat is None:
        logging.warning("No plateau found: fallback to KS-minimization")
        ks_stats = np.full(ks.shape, np.inf)
        for idx, k in enumerate(ks):
            if k < min_excess or k > max_excess:
                continue
            excess = y_pos[:k] - y_pos[k]
            if excess.size < min_excess:
                continue
            
            try:
                # floc=0 은 GPD의 위치 모수(location)를 0으로 고정
                ξ, _, σ = stats.genpareto.fit(excess, floc=0)
                if σ <= 0: continue # scale parameter는 양수여야 함
                ks_stats[idx] = stats.kstest(excess, 'genpareto', args=(ξ, 0, σ))[0]
            except Exception as e:
                logging.debug(f"GPD fit failed for k={k}: {e}")
                continue

        if np.all(np.isinf(ks_stats)):
             logging.error("KS-minimization also failed to find a valid k.")
             k_hat = min_excess # 최후의 수단
        else:
            best = np.nanargmin(ks_stats)
            k_hat = ks[best]
            logging.info(f"KS-min D={ks_stats[best]:.4f} at k={k_hat}")

    k_hat = int(np.clip(k_hat, min_excess, max_excess))
    k_hat = min(k_hat, y_pos.size - 2)

    u_hat = float(y_pos[k_hat])
    # 최종 gamma_hat은 원본 데이터로 다시 계산
    gamma_hat = np.mean(np.log(y_pos[:k_hat]) - np.log(y_pos[k_hat]))

    diag = HillDiag(k_grid=ks, gamma_hat=smooth, grad_norm=grad_norm, var_rel=var_rel, smooth_win=win)
    return u_hat, k_hat, gamma_hat, diag

def select_threshold(y: np.ndarray,
                     *,
                     u_override: Optional[float] = None,
                     return_diag: bool = False,
                     **hill_kw
                     ) -> Tuple[float, int, float] | Tuple[float, int, float, HillDiag]:
    """
    데이터에 대한 임계값을 선택합니다.
    - u_override: 임계값을 수동으로 지정.
    - hill_kw: _auto_hill에 전달할 파라미터.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y must be a 1-D non-empty array")

    u_auto, k_auto, g_auto, diag = _auto_hill(y, **hill_kw)

    if u_override is not None:
        u_hat = float(u_override)
        exceed = y[y > u_hat]
        k_hat = exceed.size
        
        if k_hat < 2: # CHANGELOG: gamma 추정을 위해 최소 2개 이상의 초과값 필요
            logging.warning(f"u_override ({u_override}) results in {k_hat} exceedances. Cannot estimate gamma reliably.")
            gamma_hat = 0.0
        else:
            # CHANGELOG: 표준적인 초과분포의 scale 추정 방식으로 변경 (통계적 정확성 향상)
            gamma_hat = np.mean(np.log(exceed) - np.log(u_hat))
    else:
        u_hat, k_hat, gamma_hat = u_auto, k_auto, g_auto

    if return_diag:
        return u_hat, k_hat, gamma_hat, diag
    return u_hat, k_hat, gamma_hat
