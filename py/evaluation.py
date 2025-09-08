#comparison.py
import numpy as np
import pandas as pd
from scipy import stats
from typing import Callable, List, Any, Optional, Tuple, Dict

# ---------------------------------------------------------------------
# API 타입: 외부 트리 모듈의 함수를 주입받아 의존성/이름충돌 제거
# find_leaf(tree, x) -> leaf
# get_leaves(tree)   -> List[leaf]
# leaf 속성:
#   - for LN:  lognorm_mu, lognorm_sigma, n_samples
#   - for GPD: gpd_params=(sigma, gamma), n_samples
# ---------------------------------------------------------------------

# 1) Truncated LogNormal logpdf (bulk 구간)
def ln_trunc_logpdf(y: float, mu: float, sigma: float,
                    trunc_left: float = 0.0,
                    trunc_right: Optional[float] = None) -> float:
    if y <= 0:
        return -np.inf
    cdf_l = 0.0 if trunc_left <= 0 else stats.lognorm.cdf(trunc_left, s=sigma, scale=np.exp(mu))
    cdf_r = 1.0 if trunc_right is None else stats.lognorm.cdf(trunc_right, s=sigma, scale=np.exp(mu))
    norm_const = max(cdf_r - cdf_l, 1e-12)
    return stats.lognorm.logpdf(y, s=sigma, scale=np.exp(mu)) - np.log(norm_const)

# 2) 한 점 log-likelihood
def point_loglik_ln_only(y: float, ln_leaf: Any) -> float:
    if y <= 0:
        return -np.inf
    return stats.lognorm.logpdf(y, s=ln_leaf.lognorm_sigma, scale=np.exp(ln_leaf.lognorm_mu))

def point_loglik_ln_gpd(y: float, u: float, ln_leaf: Any, gpd_leaf: Optional[Any]) -> float:
    if y <= u:
        # 🔧 수정: 바디 구간은 "무절단" logpdf 그대로 사용
        return stats.lognorm.logpdf(
            y, s=ln_leaf.lognorm_sigma, scale=np.exp(ln_leaf.lognorm_mu)
        )
    # y > u: tail
    if (gpd_leaf is None) or (getattr(gpd_leaf, "gpd_params", None) is None):
        return -np.inf
    excess = y - u
    sigma, gamma = gpd_leaf.gpd_params
    if sigma <= 0:
        return -np.inf
    # 조건부 tail 질량: 1 - F_B(u | leaf of bulk)
    log_tail_weight = np.log(
        1.0 - stats.lognorm.cdf(u, s=ln_leaf.lognorm_sigma, scale=np.exp(ln_leaf.lognorm_mu))
    )
    log_gpd = stats.genpareto.logpdf(excess, c=gamma, scale=sigma)
    return log_tail_weight + log_gpd


# 3) 모델 전체 logL / AIC / BIC
def model_loglik_aic_bic(
    tree_ln: Any,
    tree_gpd: Optional[Any],
    X_bulk: np.ndarray, y_bulk: np.ndarray,
    X_tail: np.ndarray, y_tail_raw: np.ndarray,
    u_threshold: float,
    *,
    find_leaf: Callable[[Any, np.ndarray], Any],
    get_leaves: Callable[[Any], List[Any]]
) -> Tuple[float, int, float, float]:
    """
    LN-only인 경우 tree_gpd=None로 호출.
    y_tail_raw: 원자료(초과치가 아니라 원값). 내부에서 excess=y-u 사용.
    """
    loglik = 0.0

    # bulk 구간
    for xi, yi in zip(X_bulk, y_bulk):
        ln_leaf = find_leaf(tree_ln, xi)
        if tree_gpd is None:   # LN-only
            loglik += point_loglik_ln_only(float(yi), ln_leaf)
        else:
            loglik += point_loglik_ln_gpd(float(yi), u_threshold, ln_leaf, None)

    # tail 구간
    for xi, yi in zip(X_tail, y_tail_raw):
        if tree_gpd is None:   # LN-only
            ln_leaf = find_leaf(tree_ln, xi)
            loglik += point_loglik_ln_only(float(yi), ln_leaf)
        else:
            ln_leaf  = find_leaf(tree_ln,  xi)
            gpd_leaf = find_leaf(tree_gpd, xi)
            loglik  += point_loglik_ln_gpd(float(yi), u_threshold, ln_leaf, gpd_leaf)

    # 파라미터 수 (리프당 2개씩: LN(μ,σ), GPD(σ,γ))
    n_ln_params  = 2 * len(get_leaves(tree_ln))
    n_gpd_params = 0 if tree_gpd is None else 2 * len(get_leaves(tree_gpd))
    k = n_ln_params + n_gpd_params

    n = len(y_bulk) + len(y_tail_raw)
    aic = 2 * k - 2 * loglik
    bic = np.log(max(n, 1)) * k - 2 * loglik
    return float(loglik), int(k), float(aic), float(bic)

# 4) 무조건부 VaR/ES (LN-only)
def unconditional_var_ln(
    tree_ln: Any, *,
    get_leaves: Callable[[Any], List[Any]],
    alpha: float = 0.99, n_sim: int = 1_000_00
) -> float:
    leaves = get_leaves(tree_ln)
    if not leaves:
        return np.nan
    w = np.array([max(getattr(lf, "n_samples", 0), 0) for lf in leaves], dtype=float)
    if w.sum() <= 0:
        return np.nan
    w /= w.sum()
    # 샘플링 (혼합 LN)
    sizes = np.maximum(1, np.floor(w * n_sim).astype(int))
    draws = [np.exp(lf.lognorm_mu + lf.lognorm_sigma * np.random.normal(size=s))
             for lf, s in zip(leaves, sizes)]
    samples = np.concatenate(draws)
    return float(np.quantile(samples, alpha))

def unconditional_es_ln(
    tree_ln: Any, *,
    get_leaves: Callable[[Any], List[Any]],
    alpha: float = 0.99, n_sim: int = 1_000_00
) -> float:
    leaves = get_leaves(tree_ln)
    if not leaves:
        return np.nan
    w = np.array([max(getattr(lf, "n_samples", 0), 0) for lf in leaves], dtype=float)
    if w.sum() <= 0:
        return np.nan
    w /= w.sum()
    sizes = np.maximum(1, np.floor(w * n_sim).astype(int))
    draws = [np.exp(lf.lognorm_mu + lf.lognorm_sigma * np.random.normal(size=s))
             for lf, s in zip(leaves, sizes)]
    samples = np.concatenate(draws)
    var_a = np.quantile(samples, alpha)
    tail = samples[samples >= var_a]
    return float(np.mean(tail)) if tail.size else np.nan

# 5) 무조건부 VaR/ES (LN+GPD 혼합) — 효율적 절단 LN 샘플링 + GPD 샘플링
def _ln_bulk_mass_per_leaf(leaves_ln: List[Any], u: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """각 LN 리프의 F_i(u)와 가중치 w_i(샘플비중) → p_bulk = Σ w_i F_i(u)"""
    w = np.array([max(getattr(lf, "n_samples", 0), 0) for lf in leaves_ln], dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(leaves_ln), dtype=float)
    w /= w.sum()
    F_u = np.array([
        stats.lognorm.cdf(u, s=lf.lognorm_sigma, scale=np.exp(lf.lognorm_mu))
        for lf in leaves_ln
    ], dtype=float)
    p_bulk = float(np.clip(np.sum(w * F_u), 0.0, 1.0))
    return F_u, w, p_bulk

def _sample_truncated_lognorm_per_leaf(lf: Any, u: float, n: int) -> np.ndarray:
    """각 리프에서 U~Unif(0, F(u)) 후 PPF(U)로 절단 LN 샘플 n개."""
    F_u = stats.lognorm.cdf(u, s=lf.lognorm_sigma, scale=np.exp(lf.lognorm_mu))
    F_u = float(np.clip(F_u, 1e-12, 1.0 - 1e-12))
    u_unif = np.random.uniform(low=0.0, high=F_u, size=n)
    return stats.lognorm.ppf(u_unif, s=lf.lognorm_sigma, scale=np.exp(lf.lognorm_mu))

def unconditional_var_mix(
    tree_ln: Any, tree_gpd: Any, u: float, *,
    get_leaves: Callable[[Any], List[Any]],
    alpha: float = 0.99, n_sim: int = 400_000
) -> float:
    leaves_ln  = get_leaves(tree_ln)
    leaves_gpd = get_leaves(tree_gpd)
    if (not leaves_ln) or (not leaves_gpd):
        return np.nan

    # (1) bulk 질량 p_bulk (LN 혼합만으로 계산)
    F_u, w_ln, p_bulk = _ln_bulk_mass_per_leaf(leaves_ln, u)

    if alpha <= p_bulk:
        # bulk 내부 분위수: 절단 LN 혼합에서 샘플링(역변환)
        # 리프별 조건부 가중치: w_i' ∝ w_i * F_i(u)
        weights_bulk = w_ln * F_u
        if weights_bulk.sum() <= 0:
            return u  # 극단 케이스
        weights_bulk /= weights_bulk.sum()
        counts = np.maximum(1, np.floor(weights_bulk * n_sim).astype(int))
        samples = [ _sample_truncated_lognorm_per_leaf(lf, u, n)
                    for lf, n in zip(leaves_ln, counts) ]
        samples = np.concatenate(samples)
        target_q = alpha / p_bulk
        return float(np.quantile(samples, target_q))

    # (2) tail 분위수: α_tail = (α - p_bulk) / (1 - p_bulk)
    alpha_tail = (alpha - p_bulk) / max(1.0 - p_bulk, 1e-12)
    w_g = np.array([max(getattr(lf, "n_samples", 0), 0) for lf in leaves_gpd], dtype=float)
    if w_g.sum() <= 0:
        w_g = np.ones(len(leaves_gpd), dtype=float)
    w_g /= w_g.sum()
    counts = np.maximum(1, np.floor(w_g * n_sim).astype(int))
    g_draws = []
    for lf, n in zip(leaves_gpd, counts):
        sigma, gamma = lf.gpd_params
        if sigma <= 0:
            continue
        exc = stats.genpareto.ppf(np.random.uniform(size=n), c=gamma, scale=sigma)
        g_draws.append(exc + u)
    if not g_draws:
        return np.nan
    tail_samples = np.concatenate(g_draws)
    return float(np.quantile(tail_samples, alpha_tail))

def unconditional_es_mix(
    tree_ln: Any, tree_gpd: Any, u: float, *,
    get_leaves: Callable[[Any], List[Any]],
    alpha: float = 0.99, n_sim: int = 400_000
) -> float:
    leaves_ln  = get_leaves(tree_ln)
    leaves_gpd = get_leaves(tree_gpd)
    if (not leaves_ln) or (not leaves_gpd):
        return np.nan

    # bulk 질량
    F_u, w_ln, p_bulk = _ln_bulk_mass_per_leaf(leaves_ln, u)

    # 🔧 추가: α가 테일에서 결정되고, γ≥1 리프가 양의 비중이면 ES는 ∞
    if alpha > p_bulk:
        w_g = np.array([max(getattr(lf, "n_samples", 0), 0) for lf in leaves_gpd], dtype=float)
        if w_g.sum() <= 0:
            w_g = np.ones(len(leaves_gpd), dtype=float)
        w_g /= w_g.sum()
        if any((getattr(lf, "gpd_params", (None, None))[1] is not None) and
               (lf.gpd_params[1] >= 1.0 - 1e-12) and
               (wg > 0) for lf, wg in zip(leaves_gpd, w_g)):
            return float("inf")  # ← 보고 단계에서 플래그 처리 권장

    # (이하 기존 샘플링 로직 유지)
    n_bulk = int(np.round(p_bulk * n_sim))
    n_tail = max(1, n_sim - n_bulk)

    # bulk 샘플
    weights_bulk = w_ln * F_u
    if weights_bulk.sum() > 0:
        weights_bulk /= weights_bulk.sum()
        counts_b = np.maximum(1, np.floor(weights_bulk * n_bulk).astype(int))
        b_draws = [ _sample_truncated_lognorm_per_leaf(lf, u, nb)
                    for lf, nb in zip(leaves_ln, counts_b) ]
        bulk_samples = np.concatenate(b_draws) if b_draws else np.empty(0)
    else:
        bulk_samples = np.empty(0)

    # tail 샘플
    w_g = np.array([max(getattr(lf, "n_samples", 0), 0) for lf in leaves_gpd], dtype=float)
    if w_g.sum() <= 0:
        w_g = np.ones(len(leaves_gpd), dtype=float)
    w_g /= w_g.sum()
    counts_t = np.maximum(1, np.floor(w_g * n_tail).astype(int))
    t_draws = []
    for lf, nt in zip(leaves_gpd, counts_t):
        sigma, gamma = lf.gpd_params
        if sigma <= 0:
            continue
        exc = stats.genpareto.rvs(c=gamma, scale=sigma, size=nt)
        t_draws.append(exc + u)
    tail_samples = np.concatenate(t_draws) if t_draws else np.empty(0)

    samples = np.concatenate([bulk_samples, tail_samples]) if (bulk_samples.size or tail_samples.size) else np.empty(0)
    if samples.size == 0:
        return np.nan
    var_a = np.quantile(samples, alpha)
    tail = samples[samples >= var_a]
    return float(np.mean(tail)) if tail.size else np.nan


# 6) GPD 리프별 GOF (KS & AD) — DataFrame 반환
def gpd_gof_leafwise(
    tree_gpd: Any,
    X_tail: np.ndarray, y_tail_excess: np.ndarray,   # excess = y - u
    *,
    find_leaf: Callable[[Any, np.ndarray], Any],
    get_leaves: Callable[[Any], List[Any]],
    method: str = "both"
) -> pd.DataFrame:
    leaves = get_leaves(tree_gpd)
    if not leaves:
        return pd.DataFrame(columns=["leaf_id","n","sigma","gamma","KS stat","KS p","AD stat","AD p"])

    # 각 tail 포인트를 리프에 매핑
    leaf_ids = [id(find_leaf(tree_gpd, xi)) for xi in X_tail]
    df_map = pd.DataFrame({"leaf_id": leaf_ids, "excess": y_tail_excess})

    rows = []
    for lf in leaves:
        lid = id(lf)
        ex = df_map.loc[df_map["leaf_id"] == lid, "excess"].to_numpy()
        if ex.size == 0:
            continue
        sigma, gamma = lf.gpd_params
        # 6.1 KS
        ks_stat, ks_p = np.nan, np.nan
        if method in ("ks", "both"):
            ks_stat, ks_p = stats.kstest(rvs=ex, cdf="genpareto", args=(gamma, 0, sigma))
        # 6.2 AD(Uniform 변환 후 Anderson–Darling)
        ad_stat, ad_p = np.nan, np.nan
        if method in ("ad", "both"):
            try:
                u = stats.genpareto.cdf(ex, c=gamma, scale=sigma)
                ad_stat, crit, sig = stats.anderson(u, dist="uniform")
                ad_p = float(np.interp(ad_stat, crit[::-1], (1 - sig/100)[::-1]))
            except Exception:
                # 간단 근사 (fallback)
                u = np.sort(stats.genpareto.cdf(ex, c=gamma, scale=sigma))
                n = len(u)
                i = np.arange(1, n+1)
                ad_stat = float(-n - np.sum((2*i - 1)/n * (np.log(u + 1e-12) + np.log(1 - u[::-1] + 1e-12))))
                ad_p = float(np.clip(np.exp(1.2937 - 5.709*ad_stat + 0.0186*ad_stat**2), 0, 1))

        rows.append({
            "leaf_id": lid,
            "n": int(ex.size),
            "sigma": float(sigma),
            "gamma": float(gamma),
            "KS stat": float(ks_stat),
            "KS p": float(ks_p),
            "AD stat": float(ad_stat),
            "AD p": float(ad_p),
        })
    return (pd.DataFrame(rows)
              .sort_values(["leaf_id"])
              .reset_index(drop=True))

# 7) 단일분포 적합 + 점수 + VaR/ES (비교 테이블용)
def compute_es_from_dist(dist, params, alpha=0.99, n_mc=300_000):
    var_a = dist.ppf(alpha, *params)
    samp  = dist.rvs(*params, size=n_mc)
    tail  = samp[samp >= var_a]
    return float(np.mean(tail)) if tail.size else np.nan

def fit_score_var_es(
    dist, data: np.ndarray, name: str,
    fix_loc0: bool = True, alpha: float = 0.99, n_es_mc: int = 300_000
) -> Dict[str, Any]:
    try:
        data = np.asarray(data, dtype=float)
        if fix_loc0:
            params = dist.fit(data, floc=0)
            k = len(params) - 1
        else:
            params = dist.fit(data)
            k = len(params)
        logpdf = dist.logpdf(data, *params)
        logpdf = logpdf[np.isfinite(logpdf)]
        logL = float(np.sum(logpdf))
        n = data.size
        aic = 2*k - 2*logL
        bic = np.log(max(n, 1)) * k - 2*logL
        var_a = float(dist.ppf(alpha, *params))
        es_a  = compute_es_from_dist(dist, params, alpha, n_es_mc)
        return {"Distribution": name, "k": int(k), "Log-Likelihood": logL,
                "AIC": float(aic), "BIC": float(bic),
                f"VaR {int(alpha*100)}%": var_a, f"ES  {int(alpha*100)}%": float(es_a)}
    except Exception as e:
        return {"Distribution": name, "k": np.nan, "Log-Likelihood": f"Fit failed: {e}",
                "AIC": np.nan, "BIC": np.nan,
                f"VaR {int(alpha*100)}%": np.nan, f"ES  {int(alpha*100)}%": np.nan}
