"""
Reddit Reply Network Analyzer
Builds networks with overlapping communities and better visualization.
"""

# ===== Fixed imports and global helpers =====
import os
import gc
import csv
import json
import math
import psutil  # type: ignore
import traceback
import logging
import warnings
import itertools
from datetime import datetime

import networkx as nx
from tqdm import tqdm
from multiprocessing import get_context, cpu_count
from community import community_louvain
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from karateclub import BigClam

# Optional threadpool control
try:
    from threadpoolctl import threadpool_limits  # type: ignore
except Exception:  # pragma: no cover
    threadpool_limits = None

# Configure logging (file only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("reddit_analysis.log", mode="w")],
)
logger = logging.getLogger(__name__)

# Replace any TODOs that Sonar flags with actionable notes
# NOTE: Logging is configured to write only to file to avoid noisy console output.

# Limit BLAS/OpenMP threads in workers to prevent oversubscription
_WORKER_MAX_THREADS = 1

def _set_worker_max_threads(n: int) -> None:
    global _WORKER_MAX_THREADS
    _WORKER_MAX_THREADS = max(1, int(n))

def _limit_worker_threads(max_threads: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_threads))
    if threadpool_limits is not None:
        try:
            threadpool_limits(max_threads=max_threads)
        except Exception:
            pass

def _init_worker() -> None:
    _limit_worker_threads(_WORKER_MAX_THREADS)

# ===== Numerically stable BigCLAM helpers (module-level, snake_case) =====
def _prepare_zero_index_edges_and_nodes(network: nx.Graph):
    nodes = list(network.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges_zero = [(node_to_idx[u], node_to_idx[v]) for u, v in network.edges()]
    return edges_zero, nodes, node_to_idx

def _get_connections_zero_index(edges, nn: int):
    conns = {i: [] for i in range(nn)}
    for i, j in edges:
        conns[i].append(j)
        conns[j].append(i)
    all_nodes = set(range(nn))
    nconns = {i: list(all_nodes - set(conns[i]) - {i}) for i in range(nn)}
    return conns, nconns

def _eval_factor(u: int, f, conns, nconns):
    fu = f[u]
    res = 0.0
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            dot = 100.0
        elif dot < -100:
            dot = -100.0
        prob = 1.0 - math.exp(-dot)
        if prob < 1e-12:
            prob = 1e-12
        res += math.log(prob)
    if len(nconns[u]) > 0:
        res -= float((f[nconns[u], :] @ fu).sum())
    return res

def _get_grad(u: int, f, conns, nconns, reg_param: float = 0.1):
    fu = f[u]
    grad = np.zeros(f.shape[1], dtype=float)
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            deriv = 1.0
        elif dot < -100:
            deriv = 1e10
        else:
            temp = math.exp(-dot)
            denom = 1.0 - temp
            deriv = (temp / denom) if denom > 1e-12 else 1e10
        grad += f[v] * deriv
    if len(nconns[u]) > 0:
        grad -= f[nconns[u], :].sum(axis=0)
    grad -= reg_param * fu
    return grad

def _grad_ascent(f, conns, nconns, n_iters: int = 40, lr: float = 0.01, clip_max: float = 10.0, step_show: int | None = None):
    nn, _ = f.shape
    rng = np.random.default_rng(42)
    for it in range(n_iters):
        perm = rng.permutation(nn)
        for u in perm:
            g = _get_grad(u, f, conns, nconns)
            f[u] += lr * g
            np.clip(f[u], 0.0, clip_max, out=f[u])
        if step_show and it % step_show == 0:
            scores = [_eval_factor(u, f, conns, nconns) for u in range(nn)]
            logger.debug(f"BigCLAM GradIter {it} score={sum(scores):.4f}")
    return f

def _find_optimal_k_for_network(_edges, nn: int, conns, nconns, max_k: int = 10, threshold: float = 0.05):
    scores = []
    optimal_k = 1
    rng = np.random.default_rng(42)
    for k in range(1, max_k + 1):
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=80, lr=0.01, step_show=None)
        score = sum([_eval_factor(u, F, conns, nconns) for u in range(nn)])
        scores.append(score)
        if k > 1:
            prev = scores[-2]
            improvement = 0.0 if abs(prev) < 1e-12 else (scores[-1] - prev) / abs(prev)
            if improvement < threshold:
                optimal_k = k - 1
                break
            optimal_k = k
    return optimal_k, scores

def run_professor_bigclam_on_network(network: nx.Graph, k: int | None = None, membership_threshold: float = 0.20):
    try:
        edges_zero, nodes_list, _ = _prepare_zero_index_edges_and_nodes(network)
        nn = len(nodes_list)
        if nn == 0:
            return {}
        conns, nconns = _get_connections_zero_index(edges_zero, nn)
        if not k or k <= 0:
            k, _ = _find_optimal_k_for_network(edges_zero, nn, conns, nconns, max_k=min(10, max(2, nn // 10)), threshold=0.03)
        rng = np.random.default_rng(42)
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=150, lr=0.01, step_show=30)
        communities = {}
        for idx in range(nn):
            vec = F[idx]
            exps = np.exp(vec - np.max(vec))
            probs = exps / (exps.sum() + 1e-12)
            assigned = [int(i) for i, p in enumerate(probs) if p >= membership_threshold] or [int(np.argmax(probs))]
            communities[nodes_list[idx]] = assigned
        return communities
    except Exception as e:
        logger.warning(f"Professor BigCLAM failed: {e}", exc_info=True)
        return {}

"""
Reddit Reply Network Analyzer
Builds networks with overlapping communities and better visualization.

This module contains the RedditReplyAnalyzer class and helper worker functions
for loading Reddit conversation data, preprocessing, constructing reply networks,
detecting communities (including overlapping communities with BigClam),
analyzing hubs, computing stability metrics, and exporting results.

Usage: import and call run_analysis(...) or use the RedditReplyAnalyzer class
directly in a Jupyter notebook for interactive exploration.
"""

# ===== Numerically stable BigCLAM helpers (module-level, snake_case) =====
def _prepare_zero_index_edges_and_nodes(network: nx.Graph):
    nodes = list(network.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges_zero = [(node_to_idx[u], node_to_idx[v]) for u, v in network.edges()]
    return edges_zero, nodes, node_to_idx

def _get_connections_zero_index(edges, nn: int):
    conns = {i: [] for i in range(nn)}
    for i, j in edges:
        conns[i].append(j)
        conns[j].append(i)
    all_nodes = set(range(nn))
    nconns = {i: list(all_nodes - set(conns[i]) - {i}) for i in range(nn)}
    return conns, nconns

def _eval_factor(u: int, f, conns, nconns):
    fu = f[u]
    res = 0.0
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            dot = 100.0
        elif dot < -100:
            dot = -100.0
        prob = 1.0 - math.exp(-dot)
        if prob < 1e-12:
            prob = 1e-12
        res += math.log(prob)
    if len(nconns[u]) > 0:
        res -= float((f[nconns[u], :] @ fu).sum())
    return res

def _get_grad(u: int, f, conns, nconns, reg_param: float = 0.1):
    fu = f[u]
    grad = np.zeros(f.shape[1], dtype=float)
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            deriv = 1.0
        elif dot < -100:
            deriv = 1e10
        else:
            temp = math.exp(-dot)
            denom = 1.0 - temp
            deriv = (temp / denom) if denom > 1e-12 else 1e10
        grad += f[v] * deriv
    if len(nconns[u]) > 0:
        grad -= f[nconns[u], :].sum(axis=0)
    grad -= reg_param * fu
    return grad

def _grad_ascent(f, conns, nconns, n_iters: int = 40, lr: float = 0.01, clip_max: float = 10.0, step_show: int | None = None):
    nn, _ = f.shape
    rng = np.random.default_rng(42)
    for it in range(n_iters):
        perm = rng.permutation(nn)
        for u in perm:
            g = _get_grad(u, f, conns, nconns)
            f[u] += lr * g
            np.clip(f[u], 0.0, clip_max, out=f[u])
        if step_show and it % step_show == 0:
            scores = [_eval_factor(u, f, conns, nconns) for u in range(nn)]
            logger.debug(f"BigCLAM GradIter {it} score={sum(scores):.4f}")
    return f

def _find_optimal_k_for_network(_edges, nn: int, conns, nconns, max_k: int = 10, threshold: float = 0.05):
    scores = []
    optimal_k = 1
    rng = np.random.default_rng(42)
    for k in range(1, max_k + 1):
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=80, lr=0.01, step_show=None)
        score = sum([_eval_factor(u, F, conns, nconns) for u in range(nn)])
        scores.append(score)
        if k > 1:
            prev = scores[-2]
            improvement = 0.0 if abs(prev) < 1e-12 else (scores[-1] - prev) / abs(prev)
            if improvement < threshold:
                optimal_k = k - 1
                break
            optimal_k = k
    return optimal_k, scores

def run_professor_bigclam_on_network(network: nx.Graph, k: int | None = None, membership_threshold: float = 0.20):
    try:
        edges_zero, nodes_list, _ = _prepare_zero_index_edges_and_nodes(network)
        nn = len(nodes_list)
        if nn == 0:
            return {}
        conns, nconns = _get_connections_zero_index(edges_zero, nn)
        if not k or k <= 0:
            k, _ = _find_optimal_k_for_network(edges_zero, nn, conns, nconns, max_k=min(10, max(2, nn // 10)), threshold=0.03)
        rng = np.random.default_rng(42)
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=150, lr=0.01, step_show=30)
        communities = {}
        for idx in range(nn):
            vec = F[idx]
            exps = np.exp(vec - np.max(vec))
            probs = exps / (exps.sum() + 1e-12)
            assigned = [int(i) for i, p in enumerate(probs) if p >= membership_threshold] or [int(np.argmax(probs))]
            communities[nodes_list[idx]] = assigned
        return communities
    except Exception as e:
        logger.warning(f"Professor BigCLAM failed: {e}", exc_info=True)
        return {}

"""
Reddit Reply Network Analyzer
Builds networks with overlapping communities and better visualization.

This module contains the RedditReplyAnalyzer class and helper worker functions
for loading Reddit conversation data, preprocessing, constructing reply networks,
detecting communities (including overlapping communities with BigClam),
analyzing hubs, computing stability metrics, and exporting results.

Usage: import and call run_analysis(...) or use the RedditReplyAnalyzer class
directly in a Jupyter notebook for interactive exploration.
"""

# ===== Fixed imports and global helpers =====
import os
import gc
import csv
import json
import math
import psutil  # type: ignore
import traceback
import logging
import warnings
import itertools
from datetime import datetime

import networkx as nx
from tqdm import tqdm
from multiprocessing import get_context, cpu_count
from community import community_louvain
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from karateclub import BigClam

# Optional threadpool control
try:
    from threadpoolctl import threadpool_limits  # type: ignore
except Exception:  # pragma: no cover
    threadpool_limits = None

# Configure logging (file only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("reddit_analysis.log", mode="w")],
)
logger = logging.getLogger(__name__)

# Replace any TODOs that Sonar flags with actionable notes
# NOTE: Logging is configured to write only to file to avoid noisy console output.

# Limit BLAS/OpenMP threads in workers to prevent oversubscription
_WORKER_MAX_THREADS = 1

def _set_worker_max_threads(n: int) -> None:
    global _WORKER_MAX_THREADS
    _WORKER_MAX_THREADS = max(1, int(n))

def _limit_worker_threads(max_threads: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_threads))
    if threadpool_limits is not None:
        try:
            threadpool_limits(max_threads=max_threads)
        except Exception:
            pass

def _init_worker() -> None:
    _limit_worker_threads(_WORKER_MAX_THREADS)

# ===== Numerically stable BigCLAM helpers (module-level, snake_case) =====
def _prepare_zero_index_edges_and_nodes(network: nx.Graph):
    nodes = list(network.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges_zero = [(node_to_idx[u], node_to_idx[v]) for u, v in network.edges()]
    return edges_zero, nodes, node_to_idx

def _get_connections_zero_index(edges, nn: int):
    conns = {i: [] for i in range(nn)}
    for i, j in edges:
        conns[i].append(j)
        conns[j].append(i)
    all_nodes = set(range(nn))
    nconns = {i: list(all_nodes - set(conns[i]) - {i}) for i in range(nn)}
    return conns, nconns

def _eval_factor(u: int, f, conns, nconns):
    fu = f[u]
    res = 0.0
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            dot = 100.0
        elif dot < -100:
            dot = -100.0
        prob = 1.0 - math.exp(-dot)
        if prob < 1e-12:
            prob = 1e-12
        res += math.log(prob)
    if len(nconns[u]) > 0:
        res -= float((f[nconns[u], :] @ fu).sum())
    return res

def _get_grad(u: int, f, conns, nconns, reg_param: float = 0.1):
    fu = f[u]
    grad = np.zeros(f.shape[1], dtype=float)
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            deriv = 1.0
        elif dot < -100:
            deriv = 1e10
        else:
            temp = math.exp(-dot)
            denom = 1.0 - temp
            deriv = (temp / denom) if denom > 1e-12 else 1e10
        grad += f[v] * deriv
    if len(nconns[u]) > 0:
        grad -= f[nconns[u], :].sum(axis=0)
    grad -= reg_param * fu
    return grad

def _grad_ascent(f, conns, nconns, n_iters: int = 40, lr: float = 0.01, clip_max: float = 10.0, step_show: int | None = None):
    nn, _ = f.shape
    rng = np.random.default_rng(42)
    for it in range(n_iters):
        perm = rng.permutation(nn)
        for u in perm:
            g = _get_grad(u, f, conns, nconns)
            f[u] += lr * g
            np.clip(f[u], 0.0, clip_max, out=f[u])
        if step_show and it % step_show == 0:
            scores = [_eval_factor(u, f, conns, nconns) for u in range(nn)]
            logger.debug(f"BigCLAM GradIter {it} score={sum(scores):.4f}")
    return f

def _find_optimal_k_for_network(_edges, nn: int, conns, nconns, max_k: int = 10, threshold: float = 0.05):
    scores = []
    optimal_k = 1
    rng = np.random.default_rng(42)
    for k in range(1, max_k + 1):
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=80, lr=0.01, step_show=None)
        score = sum([_eval_factor(u, F, conns, nconns) for u in range(nn)])
        scores.append(score)
        if k > 1:
            prev = scores[-2]
            improvement = 0.0 if abs(prev) < 1e-12 else (scores[-1] - prev) / abs(prev)
            if improvement < threshold:
                optimal_k = k - 1
                break
            optimal_k = k
    return optimal_k, scores

def run_professor_bigclam_on_network(network: nx.Graph, k: int | None = None, membership_threshold: float = 0.20):
    try:
        edges_zero, nodes_list, _ = _prepare_zero_index_edges_and_nodes(network)
        nn = len(nodes_list)
        if nn == 0:
            return {}
        conns, nconns = _get_connections_zero_index(edges_zero, nn)
        if not k or k <= 0:
            k, _ = _find_optimal_k_for_network(edges_zero, nn, conns, nconns, max_k=min(10, max(2, nn // 10)), threshold=0.03)
        rng = np.random.default_rng(42)
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=150, lr=0.01, step_show=30)
        communities = {}
        for idx in range(nn):
            vec = F[idx]
            exps = np.exp(vec - np.max(vec))
            probs = exps / (exps.sum() + 1e-12)
            assigned = [int(i) for i, p in enumerate(probs) if p >= membership_threshold] or [int(np.argmax(probs))]
            communities[nodes_list[idx]] = assigned
        return communities
    except Exception as e:
        logger.warning(f"Professor BigCLAM failed: {e}", exc_info=True)
        return {}

"""
Reddit Reply Network Analyzer
Builds networks with overlapping communities and better visualization.

This module contains the RedditReplyAnalyzer class and helper worker functions
for loading Reddit conversation data, preprocessing, constructing reply networks,
detecting communities (including overlapping communities with BigClam),
analyzing hubs, computing stability metrics, and exporting results.

Usage: import and call run_analysis(...) or use the RedditReplyAnalyzer class
directly in a Jupyter notebook for interactive exploration.
"""

# ===== Fixed imports and global helpers =====
import os
import gc
import csv
import json
import math
import psutil  # type: ignore
import traceback
import logging
import warnings
import itertools
from datetime import datetime

import networkx as nx
from tqdm import tqdm
from multiprocessing import get_context, cpu_count
from community import community_louvain
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from karateclub import BigClam

# Optional threadpool control
try:
    from threadpoolctl import threadpool_limits  # type: ignore
except Exception:  # pragma: no cover
    threadpool_limits = None

# Configure logging (file only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("reddit_analysis.log", mode="w")],
)
logger = logging.getLogger(__name__)

# Replace any TODOs that Sonar flags with actionable notes
# NOTE: Logging is configured to write only to file to avoid noisy console output.

# Limit BLAS/OpenMP threads in workers to prevent oversubscription
_WORKER_MAX_THREADS = 1

def _set_worker_max_threads(n: int) -> None:
    global _WORKER_MAX_THREADS
    _WORKER_MAX_THREADS = max(1, int(n))

def _limit_worker_threads(max_threads: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_threads))
    if threadpool_limits is not None:
        try:
            threadpool_limits(max_threads=max_threads)
        except Exception:
            pass

def _init_worker() -> None:
    _limit_worker_threads(_WORKER_MAX_THREADS)

# ===== Numerically stable BigCLAM helpers (module-level, snake_case) =====
def _prepare_zero_index_edges_and_nodes(network: nx.Graph):
    nodes = list(network.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges_zero = [(node_to_idx[u], node_to_idx[v]) for u, v in network.edges()]
    return edges_zero, nodes, node_to_idx

def _get_connections_zero_index(edges, nn: int):
    conns = {i: [] for i in range(nn)}
    for i, j in edges:
        conns[i].append(j)
        conns[j].append(i)
    all_nodes = set(range(nn))
    nconns = {i: list(all_nodes - set(conns[i]) - {i}) for i in range(nn)}
    return conns, nconns

def _eval_factor(u: int, f, conns, nconns):
    fu = f[u]
    res = 0.0
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            dot = 100.0
        elif dot < -100:
            dot = -100.0
        prob = 1.0 - math.exp(-dot)
        if prob < 1e-12:
            prob = 1e-12
        res += math.log(prob)
    if len(nconns[u]) > 0:
        res -= float((f[nconns[u], :] @ fu).sum())
    return res

def _get_grad(u: int, f, conns, nconns, reg_param: float = 0.1):
    fu = f[u]
    grad = np.zeros(f.shape[1], dtype=float)
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            deriv = 1.0
        elif dot < -100:
            deriv = 1e10
        else:
            temp = math.exp(-dot)
            denom = 1.0 - temp
            deriv = (temp / denom) if denom > 1e-12 else 1e10
        grad += f[v] * deriv
    if len(nconns[u]) > 0:
        grad -= f[nconns[u], :].sum(axis=0)
    grad -= reg_param * fu
    return grad

def _grad_ascent(f, conns, nconns, n_iters: int = 40, lr: float = 0.01, clip_max: float = 10.0, step_show: int | None = None):
    nn, _ = f.shape
    rng = np.random.default_rng(42)
    for it in range(n_iters):
        perm = rng.permutation(nn)
        for u in perm:
            g = _get_grad(u, f, conns, nconns)
            f[u] += lr * g
            np.clip(f[u], 0.0, clip_max, out=f[u])
        if step_show and it % step_show == 0:
            scores = [_eval_factor(u, f, conns, nconns) for u in range(nn)]
            logger.debug(f"BigCLAM GradIter {it} score={sum(scores):.4f}")
    return f

def _find_optimal_k_for_network(_edges, nn: int, conns, nconns, max_k: int = 10, threshold: float = 0.05):
    scores = []
    optimal_k = 1
    rng = np.random.default_rng(42)
    for k in range(1, max_k + 1):
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=80, lr=0.01, step_show=None)
        score = sum([_eval_factor(u, F, conns, nconns) for u in range(nn)])
        scores.append(score)
        if k > 1:
            prev = scores[-2]
            improvement = 0.0 if abs(prev) < 1e-12 else (scores[-1] - prev) / abs(prev)
            if improvement < threshold:
                optimal_k = k - 1
                break
            optimal_k = k
    return optimal_k, scores

def run_professor_bigclam_on_network(network: nx.Graph, k: int | None = None, membership_threshold: float = 0.20):
    try:
        edges_zero, nodes_list, _ = _prepare_zero_index_edges_and_nodes(network)
        nn = len(nodes_list)
        if nn == 0:
            return {}
        conns, nconns = _get_connections_zero_index(edges_zero, nn)
        if not k or k <= 0:
            k, _ = _find_optimal_k_for_network(edges_zero, nn, conns, nconns, max_k=min(10, max(2, nn // 10)), threshold=0.03)
        rng = np.random.default_rng(42)
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=150, lr=0.01, step_show=30)
        communities = {}
        for idx in range(nn):
            vec = F[idx]
            exps = np.exp(vec - np.max(vec))
            probs = exps / (exps.sum() + 1e-12)
            assigned = [int(i) for i, p in enumerate(probs) if p >= membership_threshold] or [int(np.argmax(probs))]
            communities[nodes_list[idx]] = assigned
        return communities
    except Exception as e:
        logger.warning(f"Professor BigCLAM failed: {e}", exc_info=True)
        return {}

"""
Reddit Reply Network Analyzer
Builds networks with overlapping communities and better visualization.

This module contains the RedditReplyAnalyzer class and helper worker functions
for loading Reddit conversation data, preprocessing, constructing reply networks,
detecting communities (including overlapping communities with BigClam),
analyzing hubs, computing stability metrics, and exporting results.

Usage: import and call run_analysis(...) or use the RedditReplyAnalyzer class
directly in a Jupyter notebook for interactive exploration.
"""

# ===== Fixed imports and global helpers =====
import os
import gc
import csv
import json
import math
import psutil  # type: ignore
import traceback
import logging
import warnings
import itertools
from datetime import datetime

import networkx as nx
from tqdm import tqdm
from multiprocessing import get_context, cpu_count
from community import community_louvain
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from karateclub import BigClam

# Optional threadpool control
try:
    from threadpoolctl import threadpool_limits  # type: ignore
except Exception:  # pragma: no cover
    threadpool_limits = None

# Configure logging (file only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("reddit_analysis.log", mode="w")],
)
logger = logging.getLogger(__name__)

# Replace any TODOs that Sonar flags with actionable notes
# NOTE: Logging is configured to write only to file to avoid noisy console output.

# Limit BLAS/OpenMP threads in workers to prevent oversubscription
_WORKER_MAX_THREADS = 1

def _set_worker_max_threads(n: int) -> None:
    global _WORKER_MAX_THREADS
    _WORKER_MAX_THREADS = max(1, int(n))

def _limit_worker_threads(max_threads: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_threads))
    if threadpool_limits is not None:
        try:
            threadpool_limits(max_threads=max_threads)
        except Exception:
            pass

def _init_worker() -> None:
    _limit_worker_threads(_WORKER_MAX_THREADS)

# ===== Numerically stable BigCLAM helpers (module-level, snake_case) =====
def _prepare_zero_index_edges_and_nodes(network: nx.Graph):
    nodes = list(network.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges_zero = [(node_to_idx[u], node_to_idx[v]) for u, v in network.edges()]
    return edges_zero, nodes, node_to_idx

def _get_connections_zero_index(edges, nn: int):
    conns = {i: [] for i in range(nn)}
    for i, j in edges:
        conns[i].append(j)
        conns[j].append(i)
    all_nodes = set(range(nn))
    nconns = {i: list(all_nodes - set(conns[i]) - {i}) for i in range(nn)}
    return conns, nconns

def _eval_factor(u: int, f, conns, nconns):
    fu = f[u]
    res = 0.0
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            dot = 100.0
        elif dot < -100:
            dot = -100.0
        prob = 1.0 - math.exp(-dot)
        if prob < 1e-12:
            prob = 1e-12
        res += math.log(prob)
    if len(nconns[u]) > 0:
        res -= float((f[nconns[u], :] @ fu).sum())
    return res

def _get_grad(u: int, f, conns, nconns, reg_param: float = 0.1):
    fu = f[u]
    grad = np.zeros(f.shape[1], dtype=float)
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            deriv = 1.0
        elif dot < -100:
            deriv = 1e10
        else:
            temp = math.exp(-dot)
            denom = 1.0 - temp
            deriv = (temp / denom) if denom > 1e-12 else 1e10
        grad += f[v] * deriv
    if len(nconns[u]) > 0:
        grad -= f[nconns[u], :].sum(axis=0)
    grad -= reg_param * fu
    return grad

def _grad_ascent(f, conns, nconns, n_iters: int = 40, lr: float = 0.01, clip_max: float = 10.0, step_show: int | None = None):
    nn, _ = f.shape
    rng = np.random.default_rng(42)
    for it in range(n_iters):
        perm = rng.permutation(nn)
        for u in perm:
            g = _get_grad(u, f, conns, nconns)
            f[u] += lr * g
            np.clip(f[u], 0.0, clip_max, out=f[u])
        if step_show and it % step_show == 0:
            scores = [_eval_factor(u, f, conns, nconns) for u in range(nn)]
            logger.debug(f"BigCLAM GradIter {it} score={sum(scores):.4f}")
    return f

def _find_optimal_k_for_network(_edges, nn: int, conns, nconns, max_k: int = 10, threshold: float = 0.05):
    scores = []
    optimal_k = 1
    rng = np.random.default_rng(42)
    for k in range(1, max_k + 1):
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=80, lr=0.01, step_show=None)
        score = sum([_eval_factor(u, F, conns, nconns) for u in range(nn)])
        scores.append(score)
        if k > 1:
            prev = scores[-2]
            improvement = 0.0 if abs(prev) < 1e-12 else (scores[-1] - prev) / abs(prev)
            if improvement < threshold:
                optimal_k = k - 1
                break
            optimal_k = k
    return optimal_k, scores

def run_professor_bigclam_on_network(network: nx.Graph, k: int | None = None, membership_threshold: float = 0.20):
    try:
        edges_zero, nodes_list, _ = _prepare_zero_index_edges_and_nodes(network)
        nn = len(nodes_list)
        if nn == 0:
            return {}
        conns, nconns = _get_connections_zero_index(edges_zero, nn)
        if not k or k <= 0:
            k, _ = _find_optimal_k_for_network(edges_zero, nn, conns, nconns, max_k=min(10, max(2, nn // 10)), threshold=0.03)
        rng = np.random.default_rng(42)
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=150, lr=0.01, step_show=30)
        communities = {}
        for idx in range(nn):
            vec = F[idx]
            exps = np.exp(vec - np.max(vec))
            probs = exps / (exps.sum() + 1e-12)
            assigned = [int(i) for i, p in enumerate(probs) if p >= membership_threshold] or [int(np.argmax(probs))]
            communities[nodes_list[idx]] = assigned
        return communities
    except Exception as e:
        logger.warning(f"Professor BigCLAM failed: {e}", exc_info=True)
        return {}

"""
Reddit Reply Network Analyzer
Builds networks with overlapping communities and better visualization.

This module contains the RedditReplyAnalyzer class and helper worker functions
for loading Reddit conversation data, preprocessing, constructing reply networks,
detecting communities (including overlapping communities with BigClam),
analyzing hubs, computing stability metrics, and exporting results.

Usage: import and call run_analysis(...) or use the RedditReplyAnalyzer class
directly in a Jupyter notebook for interactive exploration.
"""

# ===== Fixed imports and global helpers =====
import os
import gc
import csv
import json
import math
import psutil  # type: ignore
import traceback
import logging
import warnings
import itertools
from datetime import datetime

import networkx as nx
from tqdm import tqdm
from multiprocessing import get_context, cpu_count
from community import community_louvain
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from karateclub import BigClam

# Optional threadpool control
try:
    from threadpoolctl import threadpool_limits  # type: ignore
except Exception:  # pragma: no cover
    threadpool_limits = None

# Configure logging (file only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("reddit_analysis.log", mode="w")],
)
logger = logging.getLogger(__name__)

# Replace any TODOs that Sonar flags with actionable notes
# NOTE: Logging is configured to write only to file to avoid noisy console output.

# Limit BLAS/OpenMP threads in workers to prevent oversubscription
_WORKER_MAX_THREADS = 1

def _set_worker_max_threads(n: int) -> None:
    global _WORKER_MAX_THREADS
    _WORKER_MAX_THREADS = max(1, int(n))

def _limit_worker_threads(max_threads: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_threads))
    if threadpool_limits is not None:
        try:
            threadpool_limits(max_threads=max_threads)
        except Exception:
            pass

def _init_worker() -> None:
    _limit_worker_threads(_WORKER_MAX_THREADS)

# ===== Numerically stable BigCLAM helpers (module-level, snake_case) =====
def _prepare_zero_index_edges_and_nodes(network: nx.Graph):
    nodes = list(network.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges_zero = [(node_to_idx[u], node_to_idx[v]) for u, v in network.edges()]
    return edges_zero, nodes, node_to_idx

def _get_connections_zero_index(edges, nn: int):
    conns = {i: [] for i in range(nn)}
    for i, j in edges:
        conns[i].append(j)
        conns[j].append(i)
    all_nodes = set(range(nn))
    nconns = {i: list(all_nodes - set(conns[i]) - {i}) for i in range(nn)}
    return conns, nconns

def _eval_factor(u: int, f, conns, nconns):
    fu = f[u]
    res = 0.0
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            dot = 100.0
        elif dot < -100:
            dot = -100.0
        prob = 1.0 - math.exp(-dot)
        if prob < 1e-12:
            prob = 1e-12
        res += math.log(prob)
    if len(nconns[u]) > 0:
        res -= float((f[nconns[u], :] @ fu).sum())
    return res

def _get_grad(u: int, f, conns, nconns, reg_param: float = 0.1):
    fu = f[u]
    grad = np.zeros(f.shape[1], dtype=float)
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            deriv = 1.0
        elif dot < -100:
            deriv = 1e10
        else:
            temp = math.exp(-dot)
            denom = 1.0 - temp
            deriv = (temp / denom) if denom > 1e-12 else 1e10
        grad += f[v] * deriv
    if len(nconns[u]) > 0:
        grad -= f[nconns[u], :].sum(axis=0)
    grad -= reg_param * fu
    return grad

def _grad_ascent(f, conns, nconns, n_iters: int = 40, lr: float = 0.01, clip_max: float = 10.0, step_show: int | None = None):
    nn, _ = f.shape
    rng = np.random.default_rng(42)
    for it in range(n_iters):
        perm = rng.permutation(nn)
        for u in perm:
            g = _get_grad(u, f, conns, nconns)
            f[u] += lr * g
            np.clip(f[u], 0.0, clip_max, out=f[u])
        if step_show and it % step_show == 0:
            scores = [_eval_factor(u, f, conns, nconns) for u in range(nn)]
            logger.debug(f"BigCLAM GradIter {it} score={sum(scores):.4f}")
    return f

def _find_optimal_k_for_network(_edges, nn: int, conns, nconns, max_k: int = 10, threshold: float = 0.05):
    scores = []
    optimal_k = 1
    rng = np.random.default_rng(42)
    for k in range(1, max_k + 1):
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=80, lr=0.01, step_show=None)
        score = sum([_eval_factor(u, F, conns, nconns) for u in range(nn)])
        scores.append(score)
        if k > 1:
            prev = scores[-2]
            improvement = 0.0 if abs(prev) < 1e-12 else (scores[-1] - prev) / abs(prev)
            if improvement < threshold:
                optimal_k = k - 1
                break
            optimal_k = k
    return optimal_k, scores

def run_professor_bigclam_on_network(network: nx.Graph, k: int | None = None, membership_threshold: float = 0.20):
    try:
        edges_zero, nodes_list, _ = _prepare_zero_index_edges_and_nodes(network)
        nn = len(nodes_list)
        if nn == 0:
            return {}
        conns, nconns = _get_connections_zero_index(edges_zero, nn)
        if not k or k <= 0:
            k, _ = _find_optimal_k_for_network(edges_zero, nn, conns, nconns, max_k=min(10, max(2, nn // 10)), threshold=0.03)
        rng = np.random.default_rng(42)
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=150, lr=0.01, step_show=30)
        communities = {}
        for idx in range(nn):
            vec = F[idx]
            exps = np.exp(vec - np.max(vec))
            probs = exps / (exps.sum() + 1e-12)
            assigned = [int(i) for i, p in enumerate(probs) if p >= membership_threshold] or [int(np.argmax(probs))]
            communities[nodes_list[idx]] = assigned
        return communities
    except Exception as e:
        logger.warning(f"Professor BigCLAM failed: {e}", exc_info=True)
        return {}

"""
Reddit Reply Network Analyzer
Builds networks with overlapping communities and better visualization.

This module contains the RedditReplyAnalyzer class and helper worker functions
for loading Reddit conversation data, preprocessing, constructing reply networks,
detecting communities (including overlapping communities with BigClam),
analyzing hubs, computing stability metrics, and exporting results.

Usage: import and call run_analysis(...) or use the RedditReplyAnalyzer class
directly in a Jupyter notebook for interactive exploration.
"""

# ===== Fixed imports and global helpers =====
import os
import gc
import csv
import json
import math
import psutil  # type: ignore
import traceback
import logging
import warnings
import itertools
from datetime import datetime

import networkx as nx
from tqdm import tqdm
from multiprocessing import get_context, cpu_count
from community import community_louvain
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from karateclub import BigClam

# Optional threadpool control
try:
    from threadpoolctl import threadpool_limits  # type: ignore
except Exception:  # pragma: no cover
    threadpool_limits = None

# Configure logging (file only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("reddit_analysis.log", mode="w")],
)
logger = logging.getLogger(__name__)

# Replace any TODOs that Sonar flags with actionable notes
# NOTE: Logging is configured to write only to file to avoid noisy console output.

# Limit BLAS/OpenMP threads in workers to prevent oversubscription
_WORKER_MAX_THREADS = 1

def _set_worker_max_threads(n: int) -> None:
    global _WORKER_MAX_THREADS
    _WORKER_MAX_THREADS = max(1, int(n))

def _limit_worker_threads(max_threads: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_threads))
    if threadpool_limits is not None:
        try:
            threadpool_limits(max_threads=max_threads)
        except Exception:
            pass

def _init_worker() -> None:
    _limit_worker_threads(_WORKER_MAX_THREADS)

# ===== Numerically stable BigCLAM helpers (module-level, snake_case) =====
def _prepare_zero_index_edges_and_nodes(network: nx.Graph):
    nodes = list(network.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges_zero = [(node_to_idx[u], node_to_idx[v]) for u, v in network.edges()]
    return edges_zero, nodes, node_to_idx

def _get_connections_zero_index(edges, nn: int):
    conns = {i: [] for i in range(nn)}
    for i, j in edges:
        conns[i].append(j)
        conns[j].append(i)
    all_nodes = set(range(nn))
    nconns = {i: list(all_nodes - set(conns[i]) - {i}) for i in range(nn)}
    return conns, nconns

def _eval_factor(u: int, f, conns, nconns):
    fu = f[u]
    res = 0.0
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            dot = 100.0
        elif dot < -100:
            dot = -100.0
        prob = 1.0 - math.exp(-dot)
        if prob < 1e-12:
            prob = 1e-12
        res += math.log(prob)
    if len(nconns[u]) > 0:
        res -= float((f[nconns[u], :] @ fu).sum())
    return res

def _get_grad(u: int, f, conns, nconns, reg_param: float = 0.1):
    fu = f[u]
    grad = np.zeros(f.shape[1], dtype=float)
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            deriv = 1.0
        elif dot < -100:
            deriv = 1e10
        else:
            temp = math.exp(-dot)
            denom = 1.0 - temp
            deriv = (temp / denom) if denom > 1e-12 else 1e10
        grad += f[v] * deriv
    if len(nconns[u]) > 0:
        grad -= f[nconns[u], :].sum(axis=0)
    grad -= reg_param * fu
    return grad

def _grad_ascent(f, conns, nconns, n_iters: int = 40, lr: float = 0.01, clip_max: float = 10.0, step_show: int | None = None):
    nn, _ = f.shape
    rng = np.random.default_rng(42)
    for it in range(n_iters):
        perm = rng.permutation(nn)
        for u in perm:
            g = _get_grad(u, f, conns, nconns)
            f[u] += lr * g
            np.clip(f[u], 0.0, clip_max, out=f[u])
        if step_show and it % step_show == 0:
            scores = [_eval_factor(u, f, conns, nconns) for u in range(nn)]
            logger.debug(f"BigCLAM GradIter {it} score={sum(scores):.4f}")
    return f

def _find_optimal_k_for_network(_edges, nn: int, conns, nconns, max_k: int = 10, threshold: float = 0.05):
    scores = []
    optimal_k = 1
    rng = np.random.default_rng(42)
    for k in range(1, max_k + 1):
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=80, lr=0.01, step_show=None)
        score = sum([_eval_factor(u, F, conns, nconns) for u in range(nn)])
        scores.append(score)
        if k > 1:
            prev = scores[-2]
            improvement = 0.0 if abs(prev) < 1e-12 else (scores[-1] - prev) / abs(prev)
            if improvement < threshold:
                optimal_k = k - 1
                break
            optimal_k = k
    return optimal_k, scores

def run_professor_bigclam_on_network(network: nx.Graph, k: int | None = None, membership_threshold: float = 0.20):
    try:
        edges_zero, nodes_list, _ = _prepare_zero_index_edges_and_nodes(network)
        nn = len(nodes_list)
        if nn == 0:
            return {}
        conns, nconns = _get_connections_zero_index(edges_zero, nn)
        if not k or k <= 0:
            k, _ = _find_optimal_k_for_network(edges_zero, nn, conns, nconns, max_k=min(10, max(2, nn // 10)), threshold=0.03)
        rng = np.random.default_rng(42)
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=150, lr=0.01, step_show=30)
        communities = {}
        for idx in range(nn):
            vec = F[idx]
            exps = np.exp(vec - np.max(vec))
            probs = exps / (exps.sum() + 1e-12)
            assigned = [int(i) for i, p in enumerate(probs) if p >= membership_threshold] or [int(np.argmax(probs))]
            communities[nodes_list[idx]] = assigned
        return communities
    except Exception as e:
        logger.warning(f"Professor BigCLAM failed: {e}", exc_info=True)
        return {}

"""
Reddit Reply Network Analyzer
Builds networks with overlapping communities and better visualization.

This module contains the RedditReplyAnalyzer class and helper worker functions
for loading Reddit conversation data, preprocessing, constructing reply networks,
detecting communities (including overlapping communities with BigClam),
analyzing hubs, computing stability metrics, and exporting results.

Usage: import and call run_analysis(...) or use the RedditReplyAnalyzer class
directly in a Jupyter notebook for interactive exploration.
"""

# ===== Fixed imports and global helpers =====
import os
import gc
import csv
import json
import math
import psutil  # type: ignore
import traceback
import logging
import warnings
import itertools
from datetime import datetime

import networkx as nx
from tqdm import tqdm
from multiprocessing import get_context, cpu_count
from community import community_louvain
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from karateclub import BigClam

# Optional threadpool control
try:
    from threadpoolctl import threadpool_limits  # type: ignore
except Exception:  # pragma: no cover
    threadpool_limits = None

# Configure logging (file only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("reddit_analysis.log", mode="w")],
)
logger = logging.getLogger(__name__)

# Replace any TODOs that Sonar flags with actionable notes
# NOTE: Logging is configured to write only to file to avoid noisy console output.

# Limit BLAS/OpenMP threads in workers to prevent oversubscription
_WORKER_MAX_THREADS = 1

def _set_worker_max_threads(n: int) -> None:
    global _WORKER_MAX_THREADS
    _WORKER_MAX_THREADS = max(1, int(n))

def _limit_worker_threads(max_threads: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_threads))
    if threadpool_limits is not None:
        try:
            threadpool_limits(max_threads=max_threads)
        except Exception:
            pass

def _init_worker() -> None:
    _limit_worker_threads(_WORKER_MAX_THREADS)

# ===== Numerically stable BigCLAM helpers (module-level, snake_case) =====
def _prepare_zero_index_edges_and_nodes(network: nx.Graph):
    nodes = list(network.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges_zero = [(node_to_idx[u], node_to_idx[v]) for u, v in network.edges()]
    return edges_zero, nodes, node_to_idx

def _get_connections_zero_index(edges, nn: int):
    conns = {i: [] for i in range(nn)}
    for i, j in edges:
        conns[i].append(j)
        conns[j].append(i)
    all_nodes = set(range(nn))
    nconns = {i: list(all_nodes - set(conns[i]) - {i}) for i in range(nn)}
    return conns, nconns

def _eval_factor(u: int, f, conns, nconns):
    fu = f[u]
    res = 0.0
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            dot = 100.0
        elif dot < -100:
            dot = -100.0
        prob = 1.0 - math.exp(-dot)
        if prob < 1e-12:
            prob = 1e-12
        res += math.log(prob)
    if len(nconns[u]) > 0:
        res -= float((f[nconns[u], :] @ fu).sum())
    return res

def _get_grad(u: int, f, conns, nconns, reg_param: float = 0.1):
    fu = f[u]
    grad = np.zeros(f.shape[1], dtype=float)
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            deriv = 1.0
        elif dot < -100:
            deriv = 1e10
        else:
            temp = math.exp(-dot)
            denom = 1.0 - temp
            deriv = (temp / denom) if denom > 1e-12 else 1e10
        grad += f[v] * deriv
    if len(nconns[u]) > 0:
        grad -= f[nconns[u], :].sum(axis=0)
    grad -= reg_param * fu
    return grad

def _grad_ascent(f, conns, nconns, n_iters: int = 40, lr: float = 0.01, clip_max: float = 10.0, step_show: int | None = None):
    nn, _ = f.shape
    rng = np.random.default_rng(42)
    for it in range(n_iters):
        perm = rng.permutation(nn)
        for u in perm:
            g = _get_grad(u, f, conns, nconns)
            f[u] += lr * g
            np.clip(f[u], 0.0, clip_max, out=f[u])
        if step_show and it % step_show == 0:
            scores = [_eval_factor(u, f, conns, nconns) for u in range(nn)]
            logger.debug(f"BigCLAM GradIter {it} score={sum(scores):.4f}")
    return f

def _find_optimal_k_for_network(_edges, nn: int, conns, nconns, max_k: int = 10, threshold: float = 0.05):
    scores = []
    optimal_k = 1
    rng = np.random.default_rng(42)
    for k in range(1, max_k + 1):
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=80, lr=0.01, step_show=None)
        score = sum([_eval_factor(u, F, conns, nconns) for u in range(nn)])
        scores.append(score)
        if k > 1:
            prev = scores[-2]
            improvement = 0.0 if abs(prev) < 1e-12 else (scores[-1] - prev) / abs(prev)
            if improvement < threshold:
                optimal_k = k - 1
                break
            optimal_k = k
    return optimal_k, scores

def run_professor_bigclam_on_network(network: nx.Graph, k: int | None = None, membership_threshold: float = 0.20):
    try:
        edges_zero, nodes_list, _ = _prepare_zero_index_edges_and_nodes(network)
        nn = len(nodes_list)
        if nn == 0:
            return {}
        conns, nconns = _get_connections_zero_index(edges_zero, nn)
        if not k or k <= 0:
            k, _ = _find_optimal_k_for_network(edges_zero, nn, conns, nconns, max_k=min(10, max(2, nn // 10)), threshold=0.03)
        rng = np.random.default_rng(42)
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=150, lr=0.01, step_show=30)
        communities = {}
        for idx in range(nn):
            vec = F[idx]
            exps = np.exp(vec - np.max(vec))
            probs = exps / (exps.sum() + 1e-12)
            assigned = [int(i) for i, p in enumerate(probs) if p >= membership_threshold] or [int(np.argmax(probs))]
            communities[nodes_list[idx]] = assigned
        return communities
    except Exception as e:
        logger.warning(f"Professor BigCLAM failed: {e}", exc_info=True)
        return {}

"""
Reddit Reply Network Analyzer
Builds networks with overlapping communities and better visualization.

This module contains the RedditReplyAnalyzer class and helper worker functions
for loading Reddit conversation data, preprocessing, constructing reply networks,
detecting communities (including overlapping communities with BigClam),
analyzing hubs, computing stability metrics, and exporting results.

Usage: import and call run_analysis(...) or use the RedditReplyAnalyzer class
directly in a Jupyter notebook for interactive exploration.
"""

# ===== Fixed imports and global helpers =====
import os
import gc
import csv
import json
import math
import psutil  # type: ignore
import traceback
import logging
import warnings
import itertools
from datetime import datetime

import networkx as nx
from tqdm import tqdm
from multiprocessing import get_context, cpu_count
from community import community_louvain
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from karateclub import BigClam

# Optional threadpool control
try:
    from threadpoolctl import threadpool_limits  # type: ignore
except Exception:  # pragma: no cover
    threadpool_limits = None

# Configure logging (file only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("reddit_analysis.log", mode="w")],
)
logger = logging.getLogger(__name__)

# Replace any TODOs that Sonar flags with actionable notes
# NOTE: Logging is configured to write only to file to avoid noisy console output.

# Limit BLAS/OpenMP threads in workers to prevent oversubscription
_WORKER_MAX_THREADS = 1

def _set_worker_max_threads(n: int) -> None:
    global _WORKER_MAX_THREADS
    _WORKER_MAX_THREADS = max(1, int(n))

def _limit_worker_threads(max_threads: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_threads))
    if threadpool_limits is not None:
        try:
            threadpool_limits(max_threads=max_threads)
        except Exception:
            pass

def _init_worker() -> None:
    _limit_worker_threads(_WORKER_MAX_THREADS)

# ===== Numerically stable BigCLAM helpers (module-level, snake_case) =====
def _prepare_zero_index_edges_and_nodes(network: nx.Graph):
    nodes = list(network.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges_zero = [(node_to_idx[u], node_to_idx[v]) for u, v in network.edges()]
    return edges_zero, nodes, node_to_idx

def _get_connections_zero_index(edges, nn: int):
    conns = {i: [] for i in range(nn)}
    for i, j in edges:
        conns[i].append(j)
        conns[j].append(i)
    all_nodes = set(range(nn))
    nconns = {i: list(all_nodes - set(conns[i]) - {i}) for i in range(nn)}
    return conns, nconns

def _eval_factor(u: int, f, conns, nconns):
    fu = f[u]
    res = 0.0
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            dot = 100.0
        elif dot < -100:
            dot = -100.0
        prob = 1.0 - math.exp(-dot)
        if prob < 1e-12:
            prob = 1e-12
        res += math.log(prob)
    if len(nconns[u]) > 0:
        res -= float((f[nconns[u], :] @ fu).sum())
    return res

def _get_grad(u: int, f, conns, nconns, reg_param: float = 0.1):
    fu = f[u]
    grad = np.zeros(f.shape[1], dtype=float)
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            deriv = 1.0
        elif dot < -100:
            deriv = 1e10
        else:
            temp = math.exp(-dot)
            denom = 1.0 - temp
            deriv = (temp / denom) if denom > 1e-12 else 1e10
        grad += f[v] * deriv
    if len(nconns[u]) > 0:
        grad -= f[nconns[u], :].sum(axis=0)
    grad -= reg_param * fu
    return grad

def _grad_ascent(f, conns, nconns, n_iters: int = 40, lr: float = 0.01, clip_max: float = 10.0, step_show: int | None = None):
    nn, _ = f.shape
    rng = np.random.default_rng(42)
    for it in range(n_iters):
        perm = rng.permutation(nn)
        for u in perm:
            g = _get_grad(u, f, conns, nconns)
            f[u] += lr * g
            np.clip(f[u], 0.0, clip_max, out=f[u])
        if step_show and it % step_show == 0:
            scores = [_eval_factor(u, f, conns, nconns) for u in range(nn)]
            logger.debug(f"BigCLAM GradIter {it} score={sum(scores):.4f}")
    return f

def _find_optimal_k_for_network(_edges, nn: int, conns, nconns, max_k: int = 10, threshold: float = 0.05):
    scores = []
    optimal_k = 1
    rng = np.random.default_rng(42)
    for k in range(1, max_k + 1):
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=80, lr=0.01, step_show=None)
        score = sum([_eval_factor(u, F, conns, nconns) for u in range(nn)])
        scores.append(score)
        if k > 1:
            prev = scores[-2]
            improvement = 0.0 if abs(prev) < 1e-12 else (scores[-1] - prev) / abs(prev)
            if improvement < threshold:
                optimal_k = k - 1
                break
            optimal_k = k
    return optimal_k, scores

def run_professor_bigclam_on_network(network: nx.Graph, k: int | None = None, membership_threshold: float = 0.20):
    try:
        edges_zero, nodes_list, _ = _prepare_zero_index_edges_and_nodes(network)
        nn = len(nodes_list)
        if nn == 0:
            return {}
        conns, nconns = _get_connections_zero_index(edges_zero, nn)
        if not k or k <= 0:
            k, _ = _find_optimal_k_for_network(edges_zero, nn, conns, nconns, max_k=min(10, max(2, nn // 10)), threshold=0.03)
        rng = np.random.default_rng(42)
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=150, lr=0.01, step_show=30)
        communities = {}
        for idx in range(nn):
            vec = F[idx]
            exps = np.exp(vec - np.max(vec))
            probs = exps / (exps.sum() + 1e-12)
            assigned = [int(i) for i, p in enumerate(probs) if p >= membership_threshold] or [int(np.argmax(probs))]
            communities[nodes_list[idx]] = assigned
        return communities
    except Exception as e:
        logger.warning(f"Professor BigCLAM failed: {e}", exc_info=True)
        return {}

"""
Reddit Reply Network Analyzer
Builds networks with overlapping communities and better visualization.

This module contains the RedditReplyAnalyzer class and helper worker functions
for loading Reddit conversation data, preprocessing, constructing reply networks,
detecting communities (including overlapping communities with BigClam),
analyzing hubs, computing stability metrics, and exporting results.

Usage: import and call run_analysis(...) or use the RedditReplyAnalyzer class
directly in a Jupyter notebook for interactive exploration.
"""

# ===== Fixed imports and global helpers =====
import os
import gc
import csv
import json
import math
import psutil  # type: ignore
import traceback
import logging
import warnings
import itertools
from datetime import datetime

import networkx as nx
from tqdm import tqdm
from multiprocessing import get_context, cpu_count
from community import community_louvain
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from karateclub import BigClam

# Optional threadpool control
try:
    from threadpoolctl import threadpool_limits  # type: ignore
except Exception:  # pragma: no cover
    threadpool_limits = None

# Configure logging (file only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("reddit_analysis.log", mode="w")],
)
logger = logging.getLogger(__name__)

# Replace any TODOs that Sonar flags with actionable notes
# NOTE: Logging is configured to write only to file to avoid noisy console output.

# Limit BLAS/OpenMP threads in workers to prevent oversubscription
_WORKER_MAX_THREADS = 1

def _set_worker_max_threads(n: int) -> None:
    global _WORKER_MAX_THREADS
    _WORKER_MAX_THREADS = max(1, int(n))

def _limit_worker_threads(max_threads: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_threads))
    if threadpool_limits is not None:
        try:
            threadpool_limits(max_threads=max_threads)
        except Exception:
            pass

def _init_worker() -> None:
    _limit_worker_threads(_WORKER_MAX_THREADS)

# ===== Numerically stable BigCLAM helpers (module-level, snake_case) =====
def _prepare_zero_index_edges_and_nodes(network: nx.Graph):
    nodes = list(network.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges_zero = [(node_to_idx[u], node_to_idx[v]) for u, v in network.edges()]
    return edges_zero, nodes, node_to_idx

def _get_connections_zero_index(edges, nn: int):
    conns = {i: [] for i in range(nn)}
    for i, j in edges:
        conns[i].append(j)
        conns[j].append(i)
    all_nodes = set(range(nn))
    nconns = {i: list(all_nodes - set(conns[i]) - {i}) for i in range(nn)}
    return conns, nconns

def _eval_factor(u: int, f, conns, nconns):
    fu = f[u]
    res = 0.0
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            dot = 100.0
        elif dot < -100:
            dot = -100.0
        prob = 1.0 - math.exp(-dot)
        if prob < 1e-12:
            prob = 1e-12
        res += math.log(prob)
    if len(nconns[u]) > 0:
        res -= float((f[nconns[u], :] @ fu).sum())
    return res

def _get_grad(u: int, f, conns, nconns, reg_param: float = 0.1):
    fu = f[u]
    grad = np.zeros(f.shape[1], dtype=float)
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            deriv = 1.0
        elif dot < -100:
            deriv = 1e10
        else:
            temp = math.exp(-dot)
            denom = 1.0 - temp
            deriv = (temp / denom) if denom > 1e-12 else 1e10
        grad += f[v] * deriv
    if len(nconns[u]) > 0:
        grad -= f[nconns[u], :].sum(axis=0)
    grad -= reg_param * fu
    return grad

def _grad_ascent(f, conns, nconns, n_iters: int = 40, lr: float = 0.01, clip_max: float = 10.0, step_show: int | None = None):
    nn, _ = f.shape
    rng = np.random.default_rng(42)
    for it in range(n_iters):
        perm = rng.permutation(nn)
        for u in perm:
            g = _get_grad(u, f, conns, nconns)
            f[u] += lr * g
            np.clip(f[u], 0.0, clip_max, out=f[u])
        if step_show and it % step_show == 0:
            scores = [_eval_factor(u, f, conns, nconns) for u in range(nn)]
            logger.debug(f"BigCLAM GradIter {it} score={sum(scores):.4f}")
    return f

def _find_optimal_k_for_network(_edges, nn: int, conns, nconns, max_k: int = 10, threshold: float = 0.05):
    scores = []
    optimal_k = 1
    rng = np.random.default_rng(42)
    for k in range(1, max_k + 1):
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=80, lr=0.01, step_show=None)
        score = sum([_eval_factor(u, F, conns, nconns) for u in range(nn)])
        scores.append(score)
        if k > 1:
            prev = scores[-2]
            improvement = 0.0 if abs(prev) < 1e-12 else (scores[-1] - prev) / abs(prev)
            if improvement < threshold:
                optimal_k = k - 1
                break
            optimal_k = k
    return optimal_k, scores

def run_professor_bigclam_on_network(network: nx.Graph, k: int | None = None, membership_threshold: float = 0.20):
    try:
        edges_zero, nodes_list, _ = _prepare_zero_index_edges_and_nodes(network)
        nn = len(nodes_list)
        if nn == 0:
            return {}
        conns, nconns = _get_connections_zero_index(edges_zero, nn)
        if not k or k <= 0:
            k, _ = _find_optimal_k_for_network(edges_zero, nn, conns, nconns, max_k=min(10, max(2, nn // 10)), threshold=0.03)
        rng = np.random.default_rng(42)
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=150, lr=0.01, step_show=30)
        communities = {}
        for idx in range(nn):
            vec = F[idx]
            exps = np.exp(vec - np.max(vec))
            probs = exps / (exps.sum() + 1e-12)
            assigned = [int(i) for i, p in enumerate(probs) if p >= membership_threshold] or [int(np.argmax(probs))]
            communities[nodes_list[idx]] = assigned
        return communities
    except Exception as e:
        logger.warning(f"Professor BigCLAM failed: {e}", exc_info=True)
        return {}

"""
Reddit Reply Network Analyzer
Builds networks with overlapping communities and better visualization.

This module contains the RedditReplyAnalyzer class and helper worker functions
for loading Reddit conversation data, preprocessing, constructing reply networks,
detecting communities (including overlapping communities with BigClam),
analyzing hubs, computing stability metrics, and exporting results.

Usage: import and call run_analysis(...) or use the RedditReplyAnalyzer class
directly in a Jupyter notebook for interactive exploration.
"""

# ===== Fixed imports and global helpers =====
import os
import gc
import csv
import json
import math
import psutil  # type: ignore
import traceback
import logging
import warnings
import itertools
from datetime import datetime

import networkx as nx
from tqdm import tqdm
from multiprocessing import get_context, cpu_count
from community import community_louvain
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from karateclub import BigClam

# Optional threadpool control
try:
    from threadpoolctl import threadpool_limits  # type: ignore
except Exception:  # pragma: no cover
    threadpool_limits = None

# Configure logging (file only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("reddit_analysis.log", mode="w")],
)
logger = logging.getLogger(__name__)

# Replace any TODOs that Sonar flags with actionable notes
# NOTE: Logging is configured to write only to file to avoid noisy console output.

# Limit BLAS/OpenMP threads in workers to prevent oversubscription
_WORKER_MAX_THREADS = 1

def _set_worker_max_threads(n: int) -> None:
    global _WORKER_MAX_THREADS
    _WORKER_MAX_THREADS = max(1, int(n))

def _limit_worker_threads(max_threads: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_threads))
    if threadpool_limits is not None:
        try:
            threadpool_limits(max_threads=max_threads)
        except Exception:
            pass

def _init_worker() -> None:
    _limit_worker_threads(_WORKER_MAX_THREADS)

# ===== Numerically stable BigCLAM helpers (module-level, snake_case) =====
def _prepare_zero_index_edges_and_nodes(network: nx.Graph):
    nodes = list(network.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges_zero = [(node_to_idx[u], node_to_idx[v]) for u, v in network.edges()]
    return edges_zero, nodes, node_to_idx

def _get_connections_zero_index(edges, nn: int):
    conns = {i: [] for i in range(nn)}
    for i, j in edges:
        conns[i].append(j)
        conns[j].append(i)
    all_nodes = set(range(nn))
    nconns = {i: list(all_nodes - set(conns[i]) - {i}) for i in range(nn)}
    return conns, nconns

def _eval_factor(u: int, f, conns, nconns):
    fu = f[u]
    res = 0.0
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            dot = 100.0
        elif dot < -100:
            dot = -100.0
        prob = 1.0 - math.exp(-dot)
        if prob < 1e-12:
            prob = 1e-12
        res += math.log(prob)
    if len(nconns[u]) > 0:
        res -= float((f[nconns[u], :] @ fu).sum())
    return res

def _get_grad(u: int, f, conns, nconns, reg_param: float = 0.1):
    fu = f[u]
    grad = np.zeros(f.shape[1], dtype=float)
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            deriv = 1.0
        elif dot < -100:
            deriv = 1e10
        else:
            temp = math.exp(-dot)
            denom = 1.0 - temp
            deriv = (temp / denom) if denom > 1e-12 else 1e10
        grad += f[v] * deriv
    if len(nconns[u]) > 0:
        grad -= f[nconns[u], :].sum(axis=0)
    grad -= reg_param * fu
    return grad

def _grad_ascent(f, conns, nconns, n_iters: int = 40, lr: float = 0.01, clip_max: float = 10.0, step_show: int | None = None):
    nn, _ = f.shape
    rng = np.random.default_rng(42)
    for it in range(n_iters):
        perm = rng.permutation(nn)
        for u in perm:
            g = _get_grad(u, f, conns, nconns)
            f[u] += lr * g
            np.clip(f[u], 0.0, clip_max, out=f[u])
        if step_show and it % step_show == 0:
            scores = [_eval_factor(u, f, conns, nconns) for u in range(nn)]
            logger.debug(f"BigCLAM GradIter {it} score={sum(scores):.4f}")
    return f

def _find_optimal_k_for_network(_edges, nn: int, conns, nconns, max_k: int = 10, threshold: float = 0.05):
    scores = []
    optimal_k = 1
    rng = np.random.default_rng(42)
    for k in range(1, max_k + 1):
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=80, lr=0.01, step_show=None)
        score = sum([_eval_factor(u, F, conns, nconns) for u in range(nn)])
        scores.append(score)
        if k > 1:
            prev = scores[-2]
            improvement = 0.0 if abs(prev) < 1e-12 else (scores[-1] - prev) / abs(prev)
            if improvement < threshold:
                optimal_k = k - 1
                break
            optimal_k = k
    return optimal_k, scores

def run_professor_bigclam_on_network(network: nx.Graph, k: int | None = None, membership_threshold: float = 0.20):
    try:
        edges_zero, nodes_list, _ = _prepare_zero_index_edges_and_nodes(network)
        nn = len(nodes_list)
        if nn == 0:
            return {}
        conns, nconns = _get_connections_zero_index(edges_zero, nn)
        if not k or k <= 0:
            k, _ = _find_optimal_k_for_network(edges_zero, nn, conns, nconns, max_k=min(10, max(2, nn // 10)), threshold=0.03)
        rng = np.random.default_rng(42)
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=150, lr=0.01, step_show=30)
        communities = {}
        for idx in range(nn):
            vec = F[idx]
            exps = np.exp(vec - np.max(vec))
            probs = exps / (exps.sum() + 1e-12)
            assigned = [int(i) for i, p in enumerate(probs) if p >= membership_threshold] or [int(np.argmax(probs))]
            communities[nodes_list[idx]] = assigned
        return communities
    except Exception as e:
        logger.warning(f"Professor BigCLAM failed: {e}", exc_info=True)
        return {}

"""
Reddit Reply Network Analyzer
Builds networks with overlapping communities and better visualization.

This module contains the RedditReplyAnalyzer class and helper worker functions
for loading Reddit conversation data, preprocessing, constructing reply networks,
detecting communities (including overlapping communities with BigClam),
analyzing hubs, computing stability metrics, and exporting results.

Usage: import and call run_analysis(...) or use the RedditReplyAnalyzer class
directly in a Jupyter notebook for interactive exploration.
"""

# ===== Fixed imports and global helpers =====
import os
import gc
import csv
import json
import math
import psutil  # type: ignore
import traceback
import logging
import warnings
import itertools
from datetime import datetime

import networkx as nx
from tqdm import tqdm
from multiprocessing import get_context, cpu_count
from community import community_louvain
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from karateclub import BigClam

# Optional threadpool control
try:
    from threadpoolctl import threadpool_limits  # type: ignore
except Exception:  # pragma: no cover
    threadpool_limits = None

# Configure logging (file only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("reddit_analysis.log", mode="w")],
)
logger = logging.getLogger(__name__)

# Replace any TODOs that Sonar flags with actionable notes
# NOTE: Logging is configured to write only to file to avoid noisy console output.

# Limit BLAS/OpenMP threads in workers to prevent oversubscription
_WORKER_MAX_THREADS = 1

def _set_worker_max_threads(n: int) -> None:
    global _WORKER_MAX_THREADS
    _WORKER_MAX_THREADS = max(1, int(n))

def _limit_worker_threads(max_threads: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_threads))
    if threadpool_limits is not None:
        try:
            threadpool_limits(max_threads=max_threads)
        except Exception:
            pass

def _init_worker() -> None:
    _limit_worker_threads(_WORKER_MAX_THREADS)

# ===== Numerically stable BigCLAM helpers (module-level, snake_case) =====
def _prepare_zero_index_edges_and_nodes(network: nx.Graph):
    nodes = list(network.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges_zero = [(node_to_idx[u], node_to_idx[v]) for u, v in network.edges()]
    return edges_zero, nodes, node_to_idx

def _get_connections_zero_index(edges, nn: int):
    conns = {i: [] for i in range(nn)}
    for i, j in edges:
        conns[i].append(j)
        conns[j].append(i)
    all_nodes = set(range(nn))
    nconns = {i: list(all_nodes - set(conns[i]) - {i}) for i in range(nn)}
    return conns, nconns

def _eval_factor(u: int, f, conns, nconns):
    fu = f[u]
    res = 0.0
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            dot = 100.0
        elif dot < -100:
            dot = -100.0
        prob = 1.0 - math.exp(-dot)
        if prob < 1e-12:
            prob = 1e-12
        res += math.log(prob)
    if len(nconns[u]) > 0:
        res -= float((f[nconns[u], :] @ fu).sum())
    return res

def _get_grad(u: int, f, conns, nconns, reg_param: float = 0.1):
    fu = f[u]
    grad = np.zeros(f.shape[1], dtype=float)
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            deriv = 1.0
        elif dot < -100:
            deriv = 1e10
        else:
            temp = math.exp(-dot)
            denom = 1.0 - temp
            deriv = (temp / denom) if denom > 1e-12 else 1e10
        grad += f[v] * deriv
    if len(nconns[u]) > 0:
        grad -= f[nconns[u], :].sum(axis=0)
    grad -= reg_param * fu
    return grad

def _grad_ascent(f, conns, nconns, n_iters: int = 40, lr: float = 0.01, clip_max: float = 10.0, step_show: int | None = None):
    nn, _ = f.shape
    rng = np.random.default_rng(42)
    for it in range(n_iters):
        perm = rng.permutation(nn)
        for u in perm:
            g = _get_grad(u, f, conns, nconns)
            f[u] += lr * g
            np.clip(f[u], 0.0, clip_max, out=f[u])
        if step_show and it % step_show == 0:
            scores = [_eval_factor(u, f, conns, nconns) for u in range(nn)]
            logger.debug(f"BigCLAM GradIter {it} score={sum(scores):.4f}")
    return f

def _find_optimal_k_for_network(_edges, nn: int, conns, nconns, max_k: int = 10, threshold: float = 0.05):
    scores = []
    optimal_k = 1
    rng = np.random.default_rng(42)
    for k in range(1, max_k + 1):
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=80, lr=0.01, step_show=None)
        score = sum([_eval_factor(u, F, conns, nconns) for u in range(nn)])
        scores.append(score)
        if k > 1:
            prev = scores[-2]
            improvement = 0.0 if abs(prev) < 1e-12 else (scores[-1] - prev) / abs(prev)
            if improvement < threshold:
                optimal_k = k - 1
                break
            optimal_k = k
    return optimal_k, scores

def run_professor_bigclam_on_network(network: nx.Graph, k: int | None = None, membership_threshold: float = 0.20):
    try:
        edges_zero, nodes_list, _ = _prepare_zero_index_edges_and_nodes(network)
        nn = len(nodes_list)
        if nn == 0:
            return {}
        conns, nconns = _get_connections_zero_index(edges_zero, nn)
        if not k or k <= 0:
            k, _ = _find_optimal_k_for_network(edges_zero, nn, conns, nconns, max_k=min(10, max(2, nn // 10)), threshold=0.03)
        rng = np.random.default_rng(42)
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=150, lr=0.01, step_show=30)
        communities = {}
        for idx in range(nn):
            vec = F[idx]
            exps = np.exp(vec - np.max(vec))
            probs = exps / (exps.sum() + 1e-12)
            assigned = [int(i) for i, p in enumerate(probs) if p >= membership_threshold] or [int(np.argmax(probs))]
            communities[nodes_list[idx]] = assigned
        return communities
    except Exception as e:
        logger.warning(f"Professor BigCLAM failed: {e}", exc_info=True)
        return {}

"""
Reddit Reply Network Analyzer
Builds networks with overlapping communities and better visualization.

This module contains the RedditReplyAnalyzer class and helper worker functions
for loading Reddit conversation data, preprocessing, constructing reply networks,
detecting communities (including overlapping communities with BigClam),
analyzing hubs, computing stability metrics, and exporting results.

Usage: import and call run_analysis(...) or use the RedditReplyAnalyzer class
directly in a Jupyter notebook for interactive exploration.
"""

# ===== Fixed imports and global helpers =====
import os
import gc
import csv
import json
import math
import psutil  # type: ignore
import traceback
import logging
import warnings
import itertools
from datetime import datetime

import networkx as nx
from tqdm import tqdm
from multiprocessing import get_context, cpu_count
from community import community_louvain
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from karateclub import BigClam

# Optional threadpool control
try:
    from threadpoolctl import threadpool_limits  # type: ignore
except Exception:  # pragma: no cover
    threadpool_limits = None

# Configure logging (file only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("reddit_analysis.log", mode="w")],
)
logger = logging.getLogger(__name__)

# Replace any TODOs that Sonar flags with actionable notes
# NOTE: Logging is configured to write only to file to avoid noisy console output.

# Limit BLAS/OpenMP threads in workers to prevent oversubscription
_WORKER_MAX_THREADS = 1

def _set_worker_max_threads(n: int) -> None:
    global _WORKER_MAX_THREADS
    _WORKER_MAX_THREADS = max(1, int(n))

def _limit_worker_threads(max_threads: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_threads))
    if threadpool_limits is not None:
        try:
            threadpool_limits(max_threads=max_threads)
        except Exception:
            pass

def _init_worker() -> None:
    _limit_worker_threads(_WORKER_MAX_THREADS)

# ===== Numerically stable BigCLAM helpers (module-level, snake_case) =====
def _prepare_zero_index_edges_and_nodes(network: nx.Graph):
    nodes = list(network.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges_zero = [(node_to_idx[u], node_to_idx[v]) for u, v in network.edges()]
    return edges_zero, nodes, node_to_idx

def _get_connections_zero_index(edges, nn: int):
    conns = {i: [] for i in range(nn)}
    for i, j in edges:
        conns[i].append(j)
        conns[j].append(i)
    all_nodes = set(range(nn))
    nconns = {i: list(all_nodes - set(conns[i]) - {i}) for i in range(nn)}
    return conns, nconns

def _eval_factor(u: int, f, conns, nconns):
    fu = f[u]
    res = 0.0
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            dot = 100.0
        elif dot < -100:
            dot = -100.0
        prob = 1.0 - math.exp(-dot)
        if prob < 1e-12:
            prob = 1e-12
        res += math.log(prob)
    if len(nconns[u]) > 0:
        res -= float((f[nconns[u], :] @ fu).sum())
    return res

def _get_grad(u: int, f, conns, nconns, reg_param: float = 0.1):
    fu = f[u]
    grad = np.zeros(f.shape[1], dtype=float)
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            deriv = 1.0
        elif dot < -100:
            deriv = 1e10
        else:
            temp = math.exp(-dot)
            denom = 1.0 - temp
            deriv = (temp / denom) if denom > 1e-12 else 1e10
        grad += f[v] * deriv
    if len(nconns[u]) > 0:
        grad -= f[nconns[u], :].sum(axis=0)
    grad -= reg_param * fu
    return grad

def _grad_ascent(f, conns, nconns, n_iters: int = 40, lr: float = 0.01, clip_max: float = 10.0, step_show: int | None = None):
    nn, _ = f.shape
    rng = np.random.default_rng(42)
    for it in range(n_iters):
        perm = rng.permutation(nn)
        for u in perm:
            g = _get_grad(u, f, conns, nconns)
            f[u] += lr * g
            np.clip(f[u], 0.0, clip_max, out=f[u])
        if step_show and it % step_show == 0:
            scores = [_eval_factor(u, f, conns, nconns) for u in range(nn)]
            logger.debug(f"BigCLAM GradIter {it} score={sum(scores):.4f}")
    return f

def _find_optimal_k_for_network(_edges, nn: int, conns, nconns, max_k: int = 10, threshold: float = 0.05):
    scores = []
    optimal_k = 1
    rng = np.random.default_rng(42)
    for k in range(1, max_k + 1):
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=80, lr=0.01, step_show=None)
        score = sum([_eval_factor(u, F, conns, nconns) for u in range(nn)])
        scores.append(score)
        if k > 1:
            prev = scores[-2]
            improvement = 0.0 if abs(prev) < 1e-12 else (scores[-1] - prev) / abs(prev)
            if improvement < threshold:
                optimal_k = k - 1
                break
            optimal_k = k
    return optimal_k, scores

def run_professor_bigclam_on_network(network: nx.Graph, k: int | None = None, membership_threshold: float = 0.20):
    try:
        edges_zero, nodes_list, _ = _prepare_zero_index_edges_and_nodes(network)
        nn = len(nodes_list)
        if nn == 0:
            return {}
        conns, nconns = _get_connections_zero_index(edges_zero, nn)
        if not k or k <= 0:
            k, _ = _find_optimal_k_for_network(edges_zero, nn, conns, nconns, max_k=min(10, max(2, nn // 10)), threshold=0.03)
        rng = np.random.default_rng(42)
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=150, lr=0.01, step_show=30)
        communities = {}
        for idx in range(nn):
            vec = F[idx]
            exps = np.exp(vec - np.max(vec))
            probs = exps / (exps.sum() + 1e-12)
            assigned = [int(i) for i, p in enumerate(probs) if p >= membership_threshold] or [int(np.argmax(probs))]
            communities[nodes_list[idx]] = assigned
        return communities
    except Exception as e:
        logger.warning(f"Professor BigCLAM failed: {e}", exc_info=True)
        return {}

"""
Reddit Reply Network Analyzer
Builds networks with overlapping communities and better visualization.

This module contains the RedditReplyAnalyzer class and helper worker functions
for loading Reddit conversation data, preprocessing, constructing reply networks,
detecting communities (including overlapping communities with BigClam),
analyzing hubs, computing stability metrics, and exporting results.

Usage: import and call run_analysis(...) or use the RedditReplyAnalyzer class
directly in a Jupyter notebook for interactive exploration.
"""

# ===== Fixed imports and global helpers =====
import os
import gc
import csv
import json
import math
import psutil  # type: ignore
import traceback
import logging
import warnings
import itertools
from datetime import datetime

import networkx as nx
from tqdm import tqdm
from multiprocessing import get_context, cpu_count
from community import community_louvain
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from karateclub import BigClam

# Optional threadpool control
try:
    from threadpoolctl import threadpool_limits  # type: ignore
except Exception:  # pragma: no cover
    threadpool_limits = None

# Configure logging (file only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("reddit_analysis.log", mode="w")],
)
logger = logging.getLogger(__name__)

# Replace any TODOs that Sonar flags with actionable notes
# NOTE: Logging is configured to write only to file to avoid noisy console output.

# Limit BLAS/OpenMP threads in workers to prevent oversubscription
_WORKER_MAX_THREADS = 1

def _set_worker_max_threads(n: int) -> None:
    global _WORKER_MAX_THREADS
    _WORKER_MAX_THREADS = max(1, int(n))

def _limit_worker_threads(max_threads: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_threads))
    if threadpool_limits is not None:
        try:
            threadpool_limits(max_threads=max_threads)
        except Exception:
            pass

def _init_worker() -> None:
    _limit_worker_threads(_WORKER_MAX_THREADS)

# ===== Numerically stable BigCLAM helpers (module-level, snake_case) =====
def _prepare_zero_index_edges_and_nodes(network: nx.Graph):
    nodes = list(network.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges_zero = [(node_to_idx[u], node_to_idx[v]) for u, v in network.edges()]
    return edges_zero, nodes, node_to_idx

def _get_connections_zero_index(edges, nn: int):
    conns = {i: [] for i in range(nn)}
    for i, j in edges:
        conns[i].append(j)
        conns[j].append(i)
    all_nodes = set(range(nn))
    nconns = {i: list(all_nodes - set(conns[i]) - {i}) for i in range(nn)}
    return conns, nconns

def _eval_factor(u: int, f, conns, nconns):
    fu = f[u]
    res = 0.0
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            dot = 100.0
        elif dot < -100:
            dot = -100.0
        prob = 1.0 - math.exp(-dot)
        if prob < 1e-12:
            prob = 1e-12
        res += math.log(prob)
    if len(nconns[u]) > 0:
        res -= float((f[nconns[u], :] @ fu).sum())
    return res

def _get_grad(u: int, f, conns, nconns, reg_param: float = 0.1):
    fu = f[u]
    grad = np.zeros(f.shape[1], dtype=float)
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            deriv = 1.0
        elif dot < -100:
            deriv = 1e10
        else:
            temp = math.exp(-dot)
            denom = 1.0 - temp
            deriv = (temp / denom) if denom > 1e-12 else 1e10
        grad += f[v] * deriv
    if len(nconns[u]) > 0:
        grad -= f[nconns[u], :].sum(axis=0)
    grad -= reg_param * fu
    return grad

def _grad_ascent(f, conns, nconns, n_iters: int = 40, lr: float = 0.01, clip_max: float = 10.0, step_show: int | None = None):
    nn, _ = f.shape
    rng = np.random.default_rng(42)
    for it in range(n_iters):
        perm = rng.permutation(nn)
        for u in perm:
            g = _get_grad(u, f, conns, nconns)
            f[u] += lr * g
            np.clip(f[u], 0.0, clip_max, out=f[u])
        if step_show and it % step_show == 0:
            scores = [_eval_factor(u, f, conns, nconns) for u in range(nn)]
            logger.debug(f"BigCLAM GradIter {it} score={sum(scores):.4f}")
    return f

def _find_optimal_k_for_network(_edges, nn: int, conns, nconns, max_k: int = 10, threshold: float = 0.05):
    scores = []
    optimal_k = 1
    rng = np.random.default_rng(42)
    for k in range(1, max_k + 1):
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=80, lr=0.01, step_show=None)
        score = sum([_eval_factor(u, F, conns, nconns) for u in range(nn)])
        scores.append(score)
        if k > 1:
            prev = scores[-2]
            improvement = 0.0 if abs(prev) < 1e-12 else (scores[-1] - prev) / abs(prev)
            if improvement < threshold:
                optimal_k = k - 1
                break
            optimal_k = k
    return optimal_k, scores

def run_professor_bigclam_on_network(network: nx.Graph, k: int | None = None, membership_threshold: float = 0.20):
    try:
        edges_zero, nodes_list, _ = _prepare_zero_index_edges_and_nodes(network)
        nn = len(nodes_list)
        if nn == 0:
            return {}
        conns, nconns = _get_connections_zero_index(edges_zero, nn)
        if not k or k <= 0:
            k, _ = _find_optimal_k_for_network(edges_zero, nn, conns, nconns, max_k=min(10, max(2, nn // 10)), threshold=0.03)
        rng = np.random.default_rng(42)
        F = rng.random((nn, k)) * 0.1 + 0.01
        F = _grad_ascent(F, conns, nconns, n_iters=150, lr=0.01, step_show=30)
        communities = {}
        for idx in range(nn):
            vec = F[idx]
            exps = np.exp(vec - np.max(vec))
            probs = exps / (exps.sum() + 1e-12)
            assigned = [int(i) for i, p in enumerate(probs) if p >= membership_threshold] or [int(np.argmax(probs))]
            communities[nodes_list[idx]] = assigned
        return communities
    except Exception as e:
        logger.warning(f"Professor BigCLAM failed: {e}", exc_info=True)
        return {}

"""
Reddit Reply Network Analyzer
Builds networks with overlapping communities and better visualization.

This module contains the RedditReplyAnalyzer class and helper worker functions
for loading Reddit conversation data, preprocessing, constructing reply networks,
detecting communities (including overlapping communities with BigClam),
analyzing hubs, computing stability metrics, and exporting results.

Usage: import and call run_analysis(...) or use the RedditReplyAnalyzer class
directly in a Jupyter notebook for interactive exploration.
"""

# ===== Fixed imports and global helpers =====
import os
import gc
import csv
import json
import math
import psutil  # type: ignore
import traceback
import logging
import warnings
import itertools
from datetime import datetime

import networkx as nx
from tqdm import tqdm
from multiprocessing import get_context, cpu_count
from community import community_louvain
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from karateclub import BigClam

# Optional threadpool control
try:
    from threadpoolctl import threadpool_limits  # type: ignore
except Exception:  # pragma: no cover
    threadpool_limits = None

# Configure logging (file only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("reddit_analysis.log", mode="w")],
)
logger = logging.getLogger(__name__)

# Replace any TODOs that Sonar flags with actionable notes
# NOTE: Logging is configured to write only to file to avoid noisy console output.

# Limit BLAS/OpenMP threads in workers to prevent oversubscription
_WORKER_MAX_THREADS = 1

def _set_worker_max_threads(n: int) -> None:
    global _WORKER_MAX_THREADS
    _WORKER_MAX_THREADS = max(1, int(n))

def _limit_worker_threads(max_threads: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_threads))
    if threadpool_limits is not None:
        try:
            threadpool_limits(max_threads=max_threads)
        except Exception:
            pass

def _init_worker() -> None:
    _limit_worker_threads(_WORKER_MAX_THREADS)

# ===== Numerically stable BigCLAM helpers (module-level, snake_case) =====
def _prepare_zero_index_edges_and_nodes(network: nx.Graph):
    nodes = list(network.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges_zero = [(node_to_idx[u], node_to_idx[v]) for u, v in network.edges()]
    return edges_zero, nodes, node_to_idx

def _get_connections_zero_index(edges, nn: int):
    conns = {i: [] for i in range(nn)}
    for i, j in edges:
        conns[i].append(j)
        conns[j].append(i)
    all_nodes = set(range(nn))
    nconns = {i: list(all_nodes - set(conns[i]) - {i}) for i in range(nn)}
    return conns, nconns

def _eval_factor(u: int, f, conns, nconns):
    fu = f[u]
    res = 0.0
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            dot = 100.0
        elif dot < -100:
            dot = -100.0
        prob = 1.0 - math.exp(-dot)
        if prob < 1e-12:
            prob = 1e-12
        res += math.log(prob)
    if len(nconns[u]) > 0:
        res -= float((f[nconns[u], :] @ fu).sum())
    return res

def _get_grad(u: int, f, conns, nconns, reg_param: float = 0.1):
    fu = f[u]
    grad = np.zeros(f.shape[1], dtype=float)
    for v in conns[u]:
        dot = float(fu @ f[v])
        if dot > 100:
            deriv = 1.0
        elif dot < -100:
            deriv = 1e10
        else:
            temp = math.exp(-dot)
            denom = 1.0 - temp
            deriv = (temp / denom) if denom > 1e-12 else 1e10
        grad += f[v] * deriv
    if len(nconns[u]) > 0:
        grad -= f[nconns[u], :].sum(axis=0)
    grad -= reg_param * fu
    return grad

def _grad_ascent(f, conns, nconns, n_iters: int = 40, lr: float = 0.01, clip_max: float = 10.0, step_show: int | None = None):
    nn, _ = f.shape
    rng = np.random.default_rng(42)
    for it in range(n_iters):
    print("="*70)

    analyzer = RedditReplyAnalyzer(checkpoint_dir=checkpoint_dir, n_workers=n_workers)

    try:
        # Phase 1: Load Data
        df_posts, df_comments = analyzer.parallel_data_loading(data_path, use_checkpoints)

        if df_comments is None:
            print("\n✗ Failed to load data\n")
            return analyzer, None, None, None

        # Phase 1.5: Exploratory Data Analysis
        df_posts, df_comments = analyzer.exploratory_data_analysis(df_posts, df_comments)

        # Phase 2: Build Reply Networks for selected periods
        temporal_networks = analyzer.parallel_network_construction(
            df_comments,
            temporal_unit='month',
            min_gcc_size=min_gcc_size,
            periods_per_year=periods_per_year,
            use_checkpoints=use_checkpoints
        )

        if not temporal_networks:
            print("\n✗ Failed to build networks\n")
            return analyzer, None, None, None

        # Export network metrics as CSV
        network_metrics_df = analyzer.export_network_metrics(temporal_networks)

        # Phase 3: Community Detection (optionally overlapping)
        community_results = analyzer.parallel_community_detection(
            temporal_networks,
            use_checkpoints=use_checkpoints,
            overlapping=overlapping_communities
        )

        if not community_results:
            print("\n⚠️  No communities detected\n")
            return analyzer, None, None, network_metrics_df

        # Phase 4: Hub Analysis
        hub_evolution = analyzer.parallel_hub_analysis(community_results, use_checkpoints)

        # Phase 5: Stability Metrics
        stability_metrics = analyzer.calculate_stability_metrics(community_results, hub_evolution)

        # Phase 6: Visualization and Analysis (stability plots)
        if stability_metrics:
            analyzer.plot_stability_metrics(stability_metrics)

        # Export final results and generate report
        analyzer.export_results(community_results, hub_evolution, stability_metrics)
        analyzer.generate_report(stability_metrics)

        print("\n" + "="*70)
        print("✅ ENHANCED ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\n📁 Results available in:")
        print("   • community_memberships.csv")

        # Return collected artifacts on success
        return analyzer, community_results, stability_metrics, network_metrics_df

    except Exception as e:
        # Ensure any unexpected errors are logged and the function returns a safe tuple
        logger.error(f"run_analysis failed: {e}", exc_info=True)
        print(f"\n✗ Analysis failed: {e}\n")
        traceback.print_exc()
        return analyzer, None, None, None

# Usage example in Jupyter notebook:
def demo_visualizations(analyzer, community_results, hub_evolution, stability_metrics):
    """
    Demo helper showing how to call visualization functions interactively.

    Args:
        analyzer (RedditReplyAnalyzer): An initialized analyzer instance.
        community_results (dict): Output of community detection per period.
        hub_evolution (dict): Hub diagnostics per period.
        stability_metrics (dict): Stability metrics to plot and inspect.
    """
    # 1. Visualize a specific period (choose middle period by default)
    print("1. 📊 Visualizing a specific network period:")
    periods = list(community_results.keys())
    if periods:
        sample_period = periods[len(periods)//2]  # Middle period
        analyzer.visualize_network_period(sample_period, community_results, hub_evolution)

    # 2. Compare two periods
    print("\n2. 🔄 Comparing period transitions:")
    if len(periods) >= 2:
        analyzer.compare_period_transition(periods[0], periods[1], community_results, hub_evolution)

    # 3. Plot stability metrics
    print("\n3. 📈 Plotting stability metrics:")
    if stability_metrics:
        analyzer.plot_stability_metrics(stability_metrics)

    # 4. Show network metrics (example)
    print("\n4. 📋 Network metrics summary:")
    network_metrics_df = analyzer.export_network_metrics(
        {p: community_results[p]['network'] for p in community_results.keys()}
    )
    if network_metrics_df is not None:
        print(network_metrics_df.head())


if __name__ == "__main__":
    import sys

    # Use 'spawn' on Windows and in some notebook environments to avoid forking issues
    if sys.platform.startswith('win') or 'ipykernel' in sys.modules:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError as e:
            # mp.set_start_method raises RuntimeError if the start method was already set
            logger.info(f"Multiprocessing start method already set: {e}")

    # Example usage (update data_path as needed)
    analyzer, community_results, stability_metrics, network_metrics_df = run_analysis(
        data_path="../Documentaries.corpus",
        checkpoint_dir="checkpoints",
        min_gcc_size=15000,
        periods_per_year=6,
        overlapping_communities=True,  # Set to False for non-overlapping
        use_checkpoints=True
    )

# RUN IN JUPYTER NOTEBOOK:
# # Run the analysis
# analyzer, communities, stability, metrics_df = run_analysis(
#     data_path="your_data_path",
#     overlapping_communities=True
# )

# # MANUAL INSPECTION AND VISUALIZATION
# analyzer.visualize_network_period('2018-03', communities, hubs)
# analyzer.compare_period_transition('2018-03', '2018-04', communities, hubs)

# Each edge in the networks can be accessed via:
# temporal_networks['2018-03']['reply_network'].edges(data=True)
# Example usage:
# for u, v, data in G.edges(data=True):
#     print(f"Edge {u} -> {v}:")
#     print(f"  Interactions: {data['weight']}")
#     print(f"  Average sentiment: {data['avg_sentiment']:.3f}")
#     print(f"  Sentiment range: [{data['min_sentiment']:.3f}, {data['max_sentiment']:.3f}]")
#     print(f"  Sentiment std: {data['sentiment_std']:.3f}")

# # After loading data:
# df_comments = analyzer._load_comments_smart(data_path)

# # Check some sample texts
# print("\nSample texts from loaded data:")
# for _, row in df_comments.sample(n=3, random_state=42).iterrows():
#     print(f"\nUser: {row['user']}")
#     print(f"Text length: {len(row['text'])}")
#     print(f"Text preview: {row['text'][:200]}...")

# # Check overlapping communities
# period = '2018-03'
# user_communities = communities[period]['communities']
# overlapping_users = [user for user, comms in user_communities.items() if len(comms) > 1]
# print(f"Users in multiple communities: {len(overlapping_users)}")

# # Load metrics for analysis
# metrics_df = pd.read_csv('network_metrics.csv')
# print(metrics_df.head())