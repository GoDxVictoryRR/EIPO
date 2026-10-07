"""Portfolio optimization module using CVXPY."""

import logging
from typing import Any, Union
import cvxpy as cp
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def optimize_enhanced(
    alpha: Union[np.ndarray, pd.Series],
    w_b: Union[np.ndarray, pd.Series],
    w_prev: Union[np.ndarray, pd.Series, None],
    Sigma_ann: np.ndarray,
    L: np.ndarray,
    sectors: Union[pd.Series, list[str]],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve enhanced indexing portfolio optimization problem."""
    alpha_arr = np.asarray(alpha, dtype=float)
    w_b_arr = np.asarray(w_b, dtype=float)
    w_prev_arr = np.asarray(w_prev, dtype=float) if w_prev is not None else w_b_arr.copy()
    n_assets = len(w_b_arr)

    opt_cfg = cfg["optimizer"]
    lambda_risk = float(opt_cfg.get("risk_aversion", 10.0))
    te_max = float(opt_cfg.get("te_max", 0.03))
    max_active = float(opt_cfg.get("max_active_weight", 0.03))
    sector_dev_max = float(opt_cfg.get("sector_dev_max", 0.05))
    solver_name = str(opt_cfg.get("solver", "CLARABEL"))
    tc_bps = float(cfg["costs"].get("tc_bps", 10.0))
    kappa = tc_bps / 10000.0

    w = cp.Variable(n_assets)
    a = w - w_b_arr

    # Objective: maximize alpha^T a - (lambda/2) ||L^T a||_2^2 - kappa * ||w - w_prev||_1
    obj = cp.Maximize(
        alpha_arr @ a
        - (lambda_risk / 2.0) * cp.sum_squares(L.T @ a)
        - kappa * cp.norm1(w - w_prev_arr)
    )

    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0,
        a <= max_active,
        a >= -max_active,
        cp.sum_squares(L.T @ a) <= (te_max ** 2),
    ]

    # Sector active deviation constraints (sorted for deterministic solver constraint ordering)
    sector_list = list(sectors)
    unique_sectors = sorted(set(sector_list))
    for sec in unique_sectors:
        if sec == "Unknown":
            continue
        sec_idx = [i for i, s in enumerate(sector_list) if s == sec]
        if len(sec_idx) > 0:
            constraints.append(cp.sum(a[sec_idx]) <= sector_dev_max)
            constraints.append(cp.sum(a[sec_idx]) >= -sector_dev_max)

    problem = cp.Problem(obj, constraints)

    # Solve with primary solver, falling back to SCS if needed
    solved = False
    solver_used = solver_name
    for s_name in [solver_name, "SCS"]:
        solver_attr = getattr(cp, s_name, None)
        if solver_attr is None:
            continue
        try:
            problem.solve(solver=solver_attr)
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and w.value is not None:
                solved = True
                solver_used = s_name
                break
        except Exception as e:
            logger.warning("Solver %s failed: %s", s_name, e)

    if solved and w.value is not None:
        w_res = np.array(w.value, dtype=float).flatten()
        w_res = np.where(w_res < 0.0, 0.0, w_res)
        w_res = w_res / np.sum(w_res)
        a_res = w_res - w_b_arr
        ex_ante_te = float(np.sqrt(np.maximum(0.0, a_res @ Sigma_ann @ a_res)))
        info = {
            "status": problem.status,
            "solver": solver_used,
            "ex_ante_te": ex_ante_te,
            "objective": float(problem.value),
        }
        return w_res, info

    logger.warning("Optimizer failed or was infeasible, fallback to w_prev")
    w_fallback = np.copy(w_prev_arr)
    w_fallback = np.where(w_fallback < 0.0, 0.0, w_fallback)
    w_fallback = w_fallback / np.sum(w_fallback)
    a_fallback = w_fallback - w_b_arr
    ex_ante_te = float(np.sqrt(np.maximum(0.0, a_fallback @ Sigma_ann @ a_fallback)))
    info = {
        "status": "failed_fallback",
        "solver": None,
        "ex_ante_te": ex_ante_te,
        "objective": float("nan"),
    }
    return w_fallback, info


def optimize_mv(
    alpha: Union[np.ndarray, pd.Series],
    w_prev: Union[np.ndarray, pd.Series, None],
    Sigma_ann: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve long-only mean-variance baseline portfolio optimization problem."""
    alpha_arr = np.asarray(alpha, dtype=float)
    n_assets = len(alpha_arr)
    w_prev_arr = np.asarray(w_prev, dtype=float) if w_prev is not None else np.ones(n_assets) / n_assets

    opt_cfg = cfg["optimizer"]
    lambda_risk = float(opt_cfg.get("risk_aversion", 10.0))
    solver_name = str(opt_cfg.get("solver", "CLARABEL"))
    tc_bps = float(cfg["costs"].get("tc_bps", 10.0))
    kappa = tc_bps / 10000.0

    L = np.linalg.cholesky(Sigma_ann)
    w = cp.Variable(n_assets)

    obj = cp.Maximize(
        alpha_arr @ w
        - (lambda_risk / 2.0) * cp.sum_squares(L.T @ w)
        - kappa * cp.norm1(w - w_prev_arr)
    )

    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0,
    ]

    problem = cp.Problem(obj, constraints)

    solved = False
    solver_used = solver_name
    for s_name in [solver_name, "SCS"]:
        solver_attr = getattr(cp, s_name, None)
        if solver_attr is None:
            continue
        try:
            problem.solve(solver=solver_attr)
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and w.value is not None:
                solved = True
                solver_used = s_name
                break
        except Exception as e:
            logger.warning("Solver %s failed for MV: %s", s_name, e)

    if solved and w.value is not None:
        w_res = np.array(w.value, dtype=float).flatten()
        w_res = np.where(w_res < 0.0, 0.0, w_res)
        w_res = w_res / np.sum(w_res)
        info = {
            "status": problem.status,
            "solver": solver_used,
            "objective": float(problem.value),
        }
        return w_res, info

    logger.warning("MV Optimizer failed, fallback to w_prev")
    w_fallback = np.copy(w_prev_arr)
    w_fallback = np.where(w_fallback < 0.0, 0.0, w_fallback)
    w_fallback = w_fallback / np.sum(w_fallback)
    info = {
        "status": "failed_fallback",
        "solver": None,
        "objective": float("nan"),
    }
    return w_fallback, info
