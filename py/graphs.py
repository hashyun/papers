
import networkx as nx

def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0,
                  xcenter=0.5, pos=None, parent=None):
    """
    계층형 트리 구조로 그래프 G를 배치하기 위한 좌표를 dict로 반환.
    - G: networkx DiGraph (또는 Graph)
    - root: 루트 노드 (없으면 임의로 하나 선택)
    - width: 전체 가로폭
    - vert_gap: 세로 간격
    - vert_loc: 루트의 y좌표
    - xcenter: 루트의 x좌표(가로 중앙 위치)
    - pos: 이미 배치된 노드 좌표(재귀적 호출용)
    - parent: 내부적으로 부모 노드를 추적하기 위함

    원본 참고: https://github.com/mdipierro/nx_templates (BSD 라이선스)
    """
    if pos is None:
        pos = {}
    if root is None:
        # 루트 노드를 자동 선택(진짜 트리라면 진입 차수가 0인 노드)
        root = next(iter(nx.topological_sort(G))) if isinstance(G, nx.DiGraph) else list(G.nodes)[0]

    # 자식 노드들
    neighbors = list(G.successors(root)) if isinstance(G, nx.DiGraph) else list(G.neighbors(root))
    if parent is not None and parent in neighbors:
        neighbors.remove(parent)

    # 리프이면 좌표 할당하고 리턴
    if len(neighbors) == 0:
        pos[root] = (xcenter, vert_loc)
        return pos

    # 자식이 있으면, 자식들의 가로 폭을 분배
    dx = width / len(neighbors)
    nextx = xcenter - width/2 - dx/2
    pos[root] = (xcenter, vert_loc)

    for child in neighbors:
        nextx += dx
        pos = hierarchy_pos(G, root=child, width=dx, vert_gap=vert_gap,
                            vert_loc=vert_loc - vert_gap, xcenter=nextx,
                            pos=pos, parent=root)
    return pos


import matplotlib.pyplot as plt

def draw_l1_tree_graph_hier(node, feature_names):
    """
    l1 CART 트리를 계층형 레이아웃으로 시각화
    node: l1 트리 루트 (Nodel1)
    feature_names: 변수 이름 리스트
    """
    import networkx as nx
    graph = nx.DiGraph()
    node_id_counter = [0]

    def add_nodes_edges(n, parent_id=None, is_left=None):
        node_id = node_id_counter[0]
        node_id_counter[0] += 1

        # 리프 노드이면 레이블에 l1 파라미터 표시
        if n.is_leaf:
            label = f"Leaf(L1)\nmedian={n.median_val:.2f}\nlogN(mu={n.lognorm_mu:.2f}, sig={n.lognorm_sigma:.2f})"
        else:
            split_name = feature_names[n.split_var]
            label = f"{split_name} <= {n.split_thr:.2f}\ngain={n.gain:.2f}"


        graph.add_node(node_id, label=label)

        if parent_id is not None:
            direction = "Yes" if is_left else "No"
            graph.add_edge(parent_id, node_id, label=direction)

        if not n.is_leaf:
            add_nodes_edges(n.left, node_id, True)
            add_nodes_edges(n.right, node_id, False)

        return node_id

    # 트리 전체를 순회하며 노드/엣지 구성
    root_id = add_nodes_edges(node)

    # 이제 hierarchy_pos를 이용해 계층형 좌표 생성
    pos = hierarchy_pos(graph, root=root_id)
    labels = nx.get_node_attributes(graph, 'label')
    edge_labels = nx.get_edge_attributes(graph, 'label')

    plt.figure(figsize=(18, 10))
    nx.draw(graph, pos, with_labels=True, labels=labels,
            node_size=2500, node_color="lightblue",
            font_size=10, font_weight="bold",
            edge_color="gray", linewidths=1, arrows=True)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels,
                                 font_color="red")
    plt.title("l1 CART Tree Visualization (Hierarchical Layout)")
    plt.axis("off")
    plt.show()



import matplotlib.pyplot as plt

def draw_gpd_tree_graph_hier(node, feature_names):
    """
    GPD CART 트리를 계층형 레이아웃으로 시각화
    node: GPD 트리 루트 (NodeGPD)
    feature_names: 변수 이름 리스트
    """
    import networkx as nx
    graph = nx.DiGraph()
    node_id_counter = [0]

    def add_nodes_edges(n, parent_id=None, is_left=None):
        node_id = node_id_counter[0]
        node_id_counter[0] += 1

        # 리프 노드이면 레이블에 GPD 파라미터 표시
        if n.is_leaf:
            sigma, gamma = n.gpd_params
            label = (f"Leaf(GPD)\n"
                     f"nll={n.nll:.1f}\n"
                     f"sigma={sigma:.2f}, gamma={gamma:.2f}")
        else:
            split_name = feature_names[n.split_var]
            label = (f"{split_name} ≤ {n.split_thr:.2f}\n"
                     f"gain={n.split_gain:.2f}")

        graph.add_node(node_id, label=label)

        if parent_id is not None:
            direction = "Yes" if is_left else "No"
            graph.add_edge(parent_id, node_id, label=direction)

        if not n.is_leaf:
            add_nodes_edges(n.left, node_id, True)
            add_nodes_edges(n.right, node_id, False)

        return node_id

    # 트리 전체를 순회하며 노드/엣지 구성
    root_id = add_nodes_edges(node)

    # 이제 hierarchy_pos를 이용해 계층형 좌표 생성
    pos = hierarchy_pos(graph, root=root_id)
    labels = nx.get_node_attributes(graph, 'label')
    edge_labels = nx.get_edge_attributes(graph, 'label')

    plt.figure(figsize=(18, 10))
    nx.draw(graph, pos, with_labels=True, labels=labels,
            node_size=2500, node_color="lightblue",
            font_size=10, font_weight="bold",
            edge_color="gray", linewidths=1, arrows=True)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels,
                                 font_color="red")
    plt.title("GPD CART Tree Visualization (Hierarchical Layout)")
    plt.axis("off")
    plt.show()



import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from scipy.stats import ks_2samp, chi2
from scipy.optimize import minimize
import pandas as pd

# 전역: 영어 폰트 사용 및 음수 부호 깨짐 방지
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.unicode_minus'] = False

# -------------------------
# 0. 도우미 함수 및 GP 로그우도 함수
# -------------------------
def neg_log_likelihood_gpd(params, y):
    sigma, gamma = params
    if sigma <= 0:
        return 1e15
    z = 1.0 + (gamma * y / sigma)
    if np.any(z <= 0):
        return 1e15
    ll = -np.log(sigma) - (1.0/gamma + 1.0)*np.log(z)
    return -np.sum(ll)

def gp_log_likelihood(params, y):
    """GP 분포의 log 우도"""
    return -neg_log_likelihood_gpd(params, y)

# 부트스트랩 신뢰구간 함수 (QQ Plot용)
def bootstrap_ci(y_leaf, ps, B=1000):
    """
    y_leaf: 리프 데이터 (1차원 array)
    ps: 분위수 비율 (예: np.linspace(0.01, 0.99, 50))
    B: 부트스트랩 반복 횟수
    """
    boot_samples = np.zeros((B, len(ps)))
    for b in range(B):
        sample = np.random.choice(y_leaf, size=len(y_leaf), replace=True)
        boot_samples[b, :] = np.percentile(sample, ps * 100)
    lo = np.percentile(boot_samples, 2.5, axis=0)
    hi = np.percentile(boot_samples, 97.5, axis=0)
    return lo, hi

# -------------------------
# 1. Hill Plot (영문, 로그 스케일, 안정구간 음영 표시)
# -------------------------
def plot_hill_plot(y, k_min=10, k_max=None, k_target=1000):
    y_desc = np.sort(y)[::-1]  # 내림차순 정렬
    n = len(y_desc)
    if k_max is None:
        k_max = min(n - 1, n // 2)

    k_values = np.arange(k_min, k_max)
    hill_estimates = [np.mean(np.log(y_desc[:k]) - np.log(y_desc[k])) for k in k_values]

    plt.figure(figsize=(8, 5))
    # 로그 스케일 x축 사용
    plt.semilogx(k_values, hill_estimates, marker='o', label="Hill Estimator")
    # 음영으로 안정 구간을 표시 (전체 추정치에 음영 처리)
    plt.fill_between(k_values, hill_estimates, alpha=0.15)
    plt.xlabel("k (number of top order statistics)")
    plt.ylabel("Hill estimator")
    plt.title("Hill Plot for Tail Index Selection")

    if k_target < n:
        # 인덱스를 일관적으로 수정: y_desc[k_target] 사용
        u_est = y_desc[k_target]
        plt.axvline(k_target, color='red', ls='--', label=f"k_target = {k_target}")
        est_target = np.mean(np.log(y_desc[:k_target]) - np.log(u_est))
        plt.axhline(est_target, color='red', ls='--', label=f"u ≈ {u_est:.0f}")
    plt.legend()
    plt.show()



# -------------------------
# 2. GP 트리 적합도 및 비교검정
# -------------------------

# (2-1) GP 트리 리프로 데이터를 할당하는 함수
def assign_data_to_leaves_gpd(gpd_tree, X, y):
    """
    gpd_tree: 학습된 GP 트리 (NodeGPD 구조)
    X: 극단치 데이터에 해당하는 feature 행렬 (각 행이 하나의 관측치)
    y: 대응하는 초과치 (y > 0)

    각 관측치를 gpd_tree를 따라 리프로 할당하여,
    {leaf_id: y_array} 형태의 딕셔너리를 반환합니다.
    """
    # 트리 내 모든 리프 노드 수집 (재귀적 탐색)
    leaves = []
    def traverse(node):
        if node.is_leaf:
            leaves.append(node)
        else:
            traverse(node.left)
            traverse(node.right)
    traverse(gpd_tree)

    leaf_data = {i: [] for i in range(len(leaves))}
    # 각 리프에 인덱스 매핑
    leaf_map = {leaf: i for i, leaf in enumerate(leaves)}

    # 각 관측치에 대해 해당 리프를 찾음 (find_leaf_gpd 함수 사용)
    for i in range(len(X)):
        leaf = find_leaf_gpd(gpd_tree, X[i])
        leaf_id = leaf_map[leaf]
        leaf_data[leaf_id].append(y[i])

    # 리스트를 numpy array로 변환
    for key in leaf_data:
        leaf_data[key] = np.array(leaf_data[key])
    return leaf_data, leaves

# (2-2) GP 트리 리프별 QQ 플롯 그리기 (로그 스케일, 부트스트랩 CI 추가)
def plot_gp_tree_qq(gpd_tree, X_high, y_high):
    leaf_data, leaves = assign_data_to_leaves_gpd(gpd_tree, X_high, y_high)
    ps = np.linspace(0.01, 0.99, 50)  # 분위수 비율

    for leaf_id, y_leaf in leaf_data.items():
        if len(y_leaf) < 10:
            print(f"Leaf {leaf_id}: too few points ({len(y_leaf)}). Skipped.")
            continue

        sigma, gamma = leaves[leaf_id].gpd_params
        # 이론적 분위수 계산 (GP 분포)
        theo_q = np.where(gamma != 0,
                          sigma/gamma * ((1-ps)**(-gamma) - 1),
                          -sigma * np.log(1-ps))
        # 표본 분위수 계산
        samp_q = np.percentile(y_leaf, ps * 100)
        # 부트스트랩 신뢰구간 계산 (표본 분위수에 대한 95% CI)
        lo, hi = bootstrap_ci(y_leaf, ps, B=1000)

        plt.figure(figsize=(6, 6))
        plt.fill_between(theo_q, lo, hi, color='lightgrey', alpha=0.5, label="95% bootstrap CI")
        plt.plot(theo_q, samp_q, 'o', label="Data")
        # 45도 기준선
        lims = [min(theo_q.min(), samp_q.min()), max(theo_q.max(), samp_q.max())]
        plt.plot(lims, lims, 'r--', label="45° line")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Theoretical quantiles (log scale)")
        plt.ylabel("Sample quantiles (log scale)")
        plt.title(f"Leaf {leaf_id} (n={len(y_leaf)})\nσ = {sigma:.2f}, γ = {gamma:.2f}")
        plt.legend()
        plt.show()

# (2-3) 리프 간 KS 검정 수행 및 결과 테이블 출력 (p‑value 포맷 수정)
def compare_leaves_ks(leaf_data):
    res = []
    ids = list(leaf_data)
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            ks_stat, p = ks_2samp(leaf_data[ids[i]], leaf_data[ids[j]])
            res.append({"Leaf A": ids[i], "Leaf B": ids[j],
                        "KS‑stat": ks_stat, "p‑value": p})
    df = pd.DataFrame(res)
    # p‑value가 0인 경우 "<1e-323"로 표시, 나머지는 과학적 표기
    df["p‑value"] = df["p‑value"].apply(lambda x: "<1e-323" if x == 0 else f"{x:.1e}")
    print("Leaf 간 KS 검정 결과:")
    print(df)

# (2-4) Likelihood Ratio Test (LRT) 수행 (추가: AIC 계산 및 해석 문구 추가)
def gp_tree_lrt(gpd_tree, X_high, y_high):
    # 0 이하 값 제거
    y_high = y_high[y_high > 0]
    leaf_data, leaves = assign_data_to_leaves_gpd(gpd_tree, X_high, y_high)

    # GP 트리 로그우도 합 계산 (데이터가 있는 리프만 사용)
    ll_tree = 0.0
    n_leaves_used = 0
    for i, leaf in enumerate(leaves):
        if len(leaf_data[i]) > 0:
            ll_leaf = gp_log_likelihood(leaf.gpd_params, leaf_data[i])
            ll_tree += ll_leaf
            n_leaves_used += 1

    # 단일 GP 분포에 대해 MLE 수행 (전체 데이터)
    init = np.array([np.std(y_high), 0.1])
    bds  = [(1e-8, None), (None, None)]
    opt  = minimize(neg_log_likelihood_gpd, init, args=(y_high,),
                    method='L-BFGS-B', bounds=bds)
    ll_single = gp_log_likelihood(opt.x, y_high)

    lr = 2 * (ll_tree - ll_single)
    df_val = 2 * (n_leaves_used - 1)
    p = chi2.sf(lr, df_val)

    # AIC 계산
    k_tree = 2 * n_leaves_used  # 각 리프마다 2개의 파라미터
    k_single = 2
    aic_tree = 2 * k_tree - 2 * ll_tree
    aic_single = 2 * k_single - 2 * ll_single

    print("GP 트리 vs 단일 GP 적합 LRT 결과:")
    print(f"  GP 트리 로그우도 합: {ll_tree:.2f}")
    print(f"  단일 GP 로그우도: {ll_single:.2f}")
    print(f"  LRT statistic = {lr:.2f} (df = {df_val})")
    print(f"  p‑value = {p:.3e} -> {'reject' if p < 0.05 else 'fail to reject'} single GP model")
    print(f"  AIC (GP 트리) = {aic_tree:.2f}, AIC (단일 GP) = {aic_single:.2f}")

# -------------------------
# 3. 트리 관련 보조 함수
# -------------------------
def find_leaf_gpd(node, x_row):
    """
    재귀 대신 반복문 사용: 노드가 leaf가 될 때까지 split 변수와 임계값에 따라 이동.
    """
    while not node.is_leaf:
        if x_row[node.split_var] <= node.split_thr:
            node = node.left
        else:
            node = node.right
    return node


