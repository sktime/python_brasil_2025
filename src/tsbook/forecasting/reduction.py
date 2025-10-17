#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# copyright: (c) 2025, authored for the requesting user
# license: BSD-3-Clause-compatible grant for this file by the author
"""
Global hybrid reduction forecaster for sktime:
K-step direct heads + recursive continuation, with per-row normalization.

Now supports BOTH:
- pooled multi-series/hierarchical data (y/X with MultiIndex where last level is time)
- a single time series (y/X with a single time index)

Training (global):
- Builds K supervised datasets (one per step_ahead = 1..K).
- Each row uses a lag window of y (length L = window_length) and optional
  *concurrent* exogenous X at the target timestamp.
- A row-wise normalization *strategy* can be supplied to map each (lags, target)
  into a normalized space before model fitting. The same strategy is applied
  at prediction time (per step), with inverse-transform to the original scale.

Prediction (for requested horizon H):
- For steps 1..min(K, H): use the corresponding direct model h on the **observed**
  lag window (no predicted values fed back yet).
- For steps K+1..H: continue recursively with the trained 1-step model, rolling
  the window forward with its own predictions.
- Accepts either:
    * An absolute MultiIndex fh (matching y’s id+time), or a simple time Index
      if the training data was a single series.
    * A relative FH/array of positive ints (applied to every series id).

Normalization strategy API (efficient & flexible):
- Pass either:
    1) a **strategy**: a callable taking a `lags` vector and returning
       `(transform, inverse)` functions; or
    2) a **factory**: a zero-arg callable that returns such a strategy.
     3) a **string** shortcut: one of {"divide_mean", "subtract_mean",
         "normalize", "minmax"}.
- `transform(lags, target) -> (lags_n, target_n)`; `inverse(y_n) -> y`.

Includes `mean_window_normalizer()` factory: divides by the lag-window mean.

Notes
-----
- Univariate target only (one column series per id).
- If X is used in fit, you must pass **future X rows** at all required timestamps
  for prediction (for each id, and each requested timestamp).
- This is a from-scratch implementation; not copied from sktime or other libs.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
from pandas.tseries.frequencies import to_offset
from sklearn.base import clone, RegressorMixin
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


__all__ = [
    "ReductionForecaster",
    "make_reduction",
    "mean_window_normalizer",
    "subtract_mean_normalizer",
    "zscore_normalizer",
    "minmax_normalizer",
]


# ---------------------------------------------------------------------------
# Normalization strategy helpers
# ---------------------------------------------------------------------------


def _mean_window_transform(
    lags_in: np.ndarray, target: Optional[float], m: float
) -> Tuple[np.ndarray, Optional[float]]:
    lags_arr = np.asarray(lags_in, dtype=float)
    lags_n = lags_arr / m
    tgt_n = None if target is None else float(target) / m
    return lags_n, tgt_n


def _mean_window_inverse(y_n: float, m: float) -> float:
    return float(y_n) * m


class MeanWindowNormalizer:
    """Callable strategy that scales by the mean of each lag window."""

    def __call__(self, lags: np.ndarray) -> Tuple[Callable, Callable]:
        lags_arr = np.asarray(lags, dtype=float)
        m = float(np.nanmean(lags_arr)) if lags_arr.size else 1.0
        if not np.isfinite(m) or abs(m) < 1e-12:
            m = 1.0

        transform = partial(_mean_window_transform, m=m)
        inverse = partial(_mean_window_inverse, m=m)
        return transform, inverse


def mean_window_normalizer() -> Callable[[np.ndarray], Tuple[Callable, Callable]]:
    """Factory for a simple per-row normalizer: divide by window mean."""

    return MeanWindowNormalizer()


def _subtract_mean_transform(
    lags_in: np.ndarray, target: Optional[float], m: float
) -> Tuple[np.ndarray, Optional[float]]:
    lags_arr = np.asarray(lags_in, dtype=float)
    lags_n = lags_arr - m
    tgt_n = None if target is None else float(target) - m
    return lags_n, tgt_n


def _subtract_mean_inverse(y_n: float, m: float) -> float:
    return float(y_n) + m


class SubtractMeanNormalizer:
    """Center lag windows by subtracting the mean (per row)."""

    def __call__(self, lags: np.ndarray) -> Tuple[Callable, Callable]:
        lags_arr = np.asarray(lags, dtype=float)
        m = float(np.nanmean(lags_arr)) if lags_arr.size else 0.0
        if not np.isfinite(m):
            m = 0.0

        transform = partial(_subtract_mean_transform, m=m)
        inverse = partial(_subtract_mean_inverse, m=m)
        return transform, inverse


def subtract_mean_normalizer() -> Callable[[np.ndarray], Tuple[Callable, Callable]]:
    """Factory for per-row mean subtraction."""

    return SubtractMeanNormalizer()


def _zscore_transform(
    lags_in: np.ndarray, target: Optional[float], m: float, s: float
) -> Tuple[np.ndarray, Optional[float]]:
    lags_arr = np.asarray(lags_in, dtype=float)
    lags_n = (lags_arr - m) / s
    if target is None:
        tgt_n = None
    else:
        tgt_n = (float(target) - m) / s
    return lags_n, tgt_n


def _zscore_inverse(y_n: float, m: float, s: float) -> float:
    return float(y_n) * s + m


class ZScoreNormalizer:
    """Standardize lag windows using per-row mean and std."""

    def __call__(self, lags: np.ndarray) -> Tuple[Callable, Callable]:
        lags_arr = np.asarray(lags, dtype=float)
        m = float(np.nanmean(lags_arr)) if lags_arr.size else 0.0
        if not np.isfinite(m):
            m = 0.0
        s = float(np.nanstd(lags_arr, ddof=0)) if lags_arr.size else 1.0
        if not np.isfinite(s) or abs(s) < 1e-12:
            s = 1.0

        transform = partial(_zscore_transform, m=m, s=s)
        inverse = partial(_zscore_inverse, m=m, s=s)
        return transform, inverse


def zscore_normalizer() -> Callable[[np.ndarray], Tuple[Callable, Callable]]:
    """Factory for per-row z-score standardization."""

    return ZScoreNormalizer()


def _minmax_transform(
    lags_in: np.ndarray, target: Optional[float], lo: float, hi: float, scale: float
) -> Tuple[np.ndarray, Optional[float]]:
    lags_arr = np.asarray(lags_in, dtype=float)
    lags_n = (lags_arr - lo) / scale
    if target is None:
        tgt_n = None
    else:
        tgt_n = (float(target) - lo) / scale
    return lags_n, tgt_n


def _minmax_inverse(y_n: float, lo: float, scale: float) -> float:
    return float(y_n) * scale + lo


class MinMaxNormalizer:
    """Scale lag windows to [0, 1] range per row."""

    def __call__(self, lags: np.ndarray) -> Tuple[Callable, Callable]:
        lags_arr = np.asarray(lags, dtype=float)
        if lags_arr.size:
            lo = float(np.nanmin(lags_arr))
            hi = float(np.nanmax(lags_arr))
        else:
            lo = 0.0
            hi = 1.0
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi):
            hi = lo + 1.0

        scale = hi - lo
        if not np.isfinite(scale) or abs(scale) < 1e-12:
            scale = 1.0

        transform = partial(_minmax_transform, lo=lo, hi=hi, scale=scale)
        inverse = partial(_minmax_inverse, lo=lo, scale=scale)
        return transform, inverse


def minmax_normalizer() -> Callable[[np.ndarray], Tuple[Callable, Callable]]:
    """Factory for per-row min-max scaling."""

    return MinMaxNormalizer()


_NORMALIZATION_STRATEGY_REGISTRY = {
    "divide_mean": mean_window_normalizer,
    "subtract_mean": subtract_mean_normalizer,
    "normalize": zscore_normalizer,
    "minmax": minmax_normalizer,
}


def _resolve_normalization_strategy(ns):
    """Accept either a factory (zero-arg) or a strategy (lags->(transform, inverse))."""
    if ns is None:
        return None
    if isinstance(ns, str):
        key = ns.lower()
        if key not in _NORMALIZATION_STRATEGY_REGISTRY:
            options = sorted(_NORMALIZATION_STRATEGY_REGISTRY)
            raise ValueError(
                "Unknown normalization_strategy string. "
                f"Expected one of {options}, got '{ns}'."
            )
        ns = _NORMALIZATION_STRATEGY_REGISTRY[key]
    try:
        import inspect

        sig = inspect.signature(ns)
        required = [
            p
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            and p.default is p.empty
        ]
        if len(required) == 0:
            # zero-arg factory -> call it once to get the strategy
            return ns()
    except Exception:
        # if introspection fails, just treat as already-a-strategy
        pass
    return ns


# ---------------------------------------------------------------------------
# utils (pure-python; no sktime private imports)
# ---------------------------------------------------------------------------


def _check_regressor(estimator: RegressorMixin) -> None:
    if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
        raise TypeError("estimator must implement scikit-learn's fit/predict.")


def _as_positive_int_fh(
    arr_like: Union[Iterable[int], np.ndarray, List[int]],
) -> np.ndarray:
    """Return strictly positive integer steps, sorted and unique."""
    arr = np.asarray(list(arr_like)).reshape(-1)
    if arr.size == 0:
        raise ValueError("fh must contain at least one step ahead.")
    if not np.issubdtype(arr.dtype, np.integer):
        if np.issubdtype(arr.dtype, np.floating) and np.all(np.mod(arr, 1) == 0):
            arr = arr.astype(int)
        else:
            raise ValueError("fh must be an iterable of integers.")
    if np.any(arr < 1):
        raise ValueError("All steps in fh must be >= 1 (strictly out-of-sample).")
    return np.unique(np.sort(arr))


def _infer_freq_from_index(idx: pd.Index):
    """Best-effort frequency inference (DatetimeIndex/PeriodIndex)."""
    if isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex)):
        if idx.freq is not None:
            return idx.freq
        try:
            return pd.infer_freq(idx)
        except Exception:
            return None
    return None


def _future_index_like(idx: pd.Index, horizon: int) -> Tuple[pd.Index, bool]:
    """
    Build a future index of length `horizon` "like" `idx`.
    Returns (index, absolute?), where absolute indicates absolute time (True) or
    simple 1..H relative steps (False).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")

    if isinstance(idx, pd.DatetimeIndex):
        raw_freq = idx.freq or _infer_freq_from_index(idx)
        offset = None
        if raw_freq is not None:
            try:
                offset = to_offset(raw_freq)
            except (TypeError, ValueError):
                offset = None
        if offset is None and len(idx) >= 2:
            step = idx[-1] - idx[-2]
            try:
                offset = to_offset(step)
            except (TypeError, ValueError):
                offset = None
        if offset is not None:
            start = idx[-1] + offset
            return (
                pd.date_range(start=start, periods=horizon, freq=offset, tz=idx.tz),
                True,
            )
        return pd.RangeIndex(1, horizon + 1), False

    if isinstance(idx, pd.PeriodIndex):
        freq = idx.freq
        if freq is not None:
            start = idx[-1] + 1
            return pd.period_range(start=start, periods=horizon, freq=freq), True
        return pd.RangeIndex(1, horizon + 1), False

    if isinstance(idx, (pd.RangeIndex, pd.Index)) and is_integer_dtype(idx.dtype):
        start = idx[-1] + 1
        return pd.RangeIndex(start, start + horizon), True

    return pd.RangeIndex(1, horizon + 1), False


def _select_future_rows(
    X_future: pd.DataFrame, idx: Union[pd.Index, pd.MultiIndex], allow_fill: bool = True
) -> pd.DataFrame:
    """Select rows of X_future at index `idx`, optionally imputing missing rows."""
    if not isinstance(idx, (pd.Index, pd.MultiIndex)):
        idx = pd.Index(idx)

    if not allow_fill:
        missing = idx.difference(X_future.index)
        if len(missing) > 0:
            sample = list(missing[:3])
            raise ValueError(
                "Missing required rows in X for forecast timestamps. "
                f"Examples: {sample} (total missing: {len(missing)})."
            )
        return X_future.loc[idx]

    X_aligned = X_future.reindex(idx)
    if X_aligned.isnull().values.any():
        X_aligned = X_aligned.ffill().bfill()

    if X_aligned.isnull().values.any():
        missing_rows = X_aligned.index[X_aligned.isnull().any(axis=1)]
        sample = list(missing_rows[:3])
        raise ValueError(
            "Missing required rows in X for forecast timestamps even after fill. "
            f"Examples: {sample} (total missing: {len(missing_rows)})."
        )

    return X_aligned


def _flatten_multiindex_to_time(
    y_or_X: Union[pd.Series, pd.DataFrame], ids
) -> Union[pd.Series, pd.DataFrame]:
    """Return object with *time-only* index for a specific ids tuple."""
    if isinstance(ids, tuple):
        key = ids
    else:
        key = (ids,)
    return y_or_X.xs(key, level=list(range(y_or_X.index.nlevels - 1)))


def _iter_series_groups(y: pd.Series):
    """Yield (ids_tuple, y_single_series_with_time_index)."""
    nlvls = y.index.nlevels
    id_lvls = list(range(nlvls - 1))
    # keep order stable
    group_level = id_lvls if len(id_lvls) != 1 else id_lvls[0]
    for ids, y_g in y.groupby(level=group_level, sort=False):
        if not isinstance(ids, tuple):
            ids = (ids,)
        y_flat = y_g.droplevel(id_lvls)
        yield ids, y_flat


def _build_supervised_table_single(
    y: pd.Series,
    X: Optional[pd.DataFrame],
    window_length: int,
    steps_ahead: int,
    x_mode: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Turn (y, X) into (Xt, yt) for one series and one horizon."""
    if window_length < 1:
        raise ValueError("window_length must be >= 1.")
    if not isinstance(steps_ahead, int) or steps_ahead < 1:
        raise ValueError("steps_ahead must be a positive integer.")
    if x_mode not in ("none", "concurrent", "auto"):
        raise ValueError("x_mode must be one of {'none', 'concurrent', 'auto'}.")

    use_X = (X is not None) and (x_mode in ("concurrent", "auto"))

    values = y.to_numpy()
    n = len(values)

    # anchors t valid from window_length-1 .. n - steps_ahead - 1
    max_anchor = n - steps_ahead - 1
    if max_anchor < (window_length - 1):
        raise ValueError(
            "Not enough observations: need at least window_length + steps_ahead. "
            f"Got len(y)={n}, window_length={window_length}, steps_ahead={steps_ahead}."
        )

    rows = []
    targets = []
    t_index = []

    for t in range(window_length - 1, max_anchor + 1):
        # lags y[t], y[t-1], ..., y[t-window_length+1]  (newest first)
        lag_block = values[t : t - window_length : -1]
        if lag_block.shape[0] != window_length:
            lag_block = np.asarray(
                [values[t - i] for i in range(window_length)], dtype=float
            )

        row = {f"y_lag_{i+1}": lag_block[i] for i in range(window_length)}
        target_time = y.index[t + steps_ahead]
        y_target = values[t + steps_ahead]

        if use_X:
            if target_time not in X.index:
                # Feature placeholder; user should ensure X completeness
                for c in X.columns:
                    row[f"X_{c}"] = np.nan
            else:
                xrow = X.loc[target_time]
                if isinstance(xrow, pd.DataFrame):
                    xrow = xrow.iloc[0]
                for c in X.columns:
                    row[f"X_{c}"] = xrow[c]

        rows.append(row)
        targets.append(y_target)
        t_index.append(target_time)

    Xt = pd.DataFrame(rows, index=pd.Index(t_index, name=y.index.name))
    yt = pd.Series(targets, index=Xt.index, name=y.name)
    return Xt, yt


def _build_supervised_table_global(
    y: pd.Series,
    X: Optional[pd.DataFrame],
    window_length: int,
    steps_ahead: int,
    x_mode: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Supervised table across all ids for one horizon, stacked with MultiIndex index."""
    nlvls = y.index.nlevels
    id_lvls = list(range(nlvls - 1))
    id_names = list(y.index.names[:-1])
    time_name = y.index.names[-1]

    Xt_list = []
    yt_list = []
    idx_list = []

    # iterate ids
    for ids, y_flat in _iter_series_groups(y):
        X_flat = None
        if X is not None:
            X_flat = _flatten_multiindex_to_time(X, ids)
            if not X_flat.index.is_monotonic_increasing:
                X_flat = X_flat.sort_index()
        if not y_flat.index.is_monotonic_increasing:
            y_flat = y_flat.sort_index()

        Xt_g, yt_g = _build_supervised_table_single(
            y=y_flat,
            X=X_flat,
            window_length=window_length,
            steps_ahead=steps_ahead,
            x_mode=x_mode,
        )

        # attach ids to index -> MultiIndex (ids..., time)
        if len(ids) == 1:
            new_index = pd.MultiIndex.from_arrays(
                [[ids[0]] * len(Xt_g), Xt_g.index],
                names=id_names + [time_name],
            )
        else:
            arrays = [[ids[j]] * len(Xt_g) for j in range(len(ids))]
            arrays.append(list(Xt_g.index))
            new_index = pd.MultiIndex.from_arrays(arrays, names=id_names + [time_name])

        Xt_g.index = new_index
        yt_g.index = new_index

        Xt_list.append(Xt_g)
        yt_list.append(yt_g)

    Xt_all = pd.concat(Xt_list, axis=0)
    yt_all = pd.concat(yt_list, axis=0)

    return Xt_all.sort_index(), yt_all.sort_index()


def _normalize_supervised_rowwise(
    Xt: pd.DataFrame,
    yt: pd.Series,
    L: int,
    normalization_strategy: Optional[Callable[[np.ndarray], Tuple[Callable, Callable]]],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply per-row normalization to (y-lags, target)."""
    if normalization_strategy is None:
        return Xt, yt

    Xt_out = Xt.copy()
    yt_out = yt.astype(float, copy=True)
    lag_cols = [f"y_lag_{i+1}" for i in range(L)]
    lag_idx = [Xt_out.columns.get_loc(col) for col in lag_cols]

    for i in range(len(Xt_out)):
        lags = Xt_out.iloc[i, lag_idx].to_numpy(dtype=float)
        target_value = yt_out.iloc[i]
        transform, _ = normalization_strategy(lags)  # per-window
        lags_n, tgt_n = transform(lags, target_value)
        Xt_out.iloc[i, lag_idx] = lags_n
        if tgt_n is not None:
            yt_out.iloc[i] = float(tgt_n)

    return Xt_out, yt_out


def _make_group_future_multiindex(
    ids: Tuple, future_time_index: pd.Index, id_names: List[str], time_name: str
) -> pd.MultiIndex:
    """Build a MultiIndex combining ids (tuple) and per-group future time index."""
    arrays = [[ids[j]] * len(future_time_index) for j in range(len(ids))]
    arrays.append(list(future_time_index))
    return pd.MultiIndex.from_arrays(arrays, names=id_names + [time_name])


def _steps_and_full_future_for_group(
    train_time_index: pd.Index,
    req_times: Optional[pd.Index] = None,
    rel_steps: Optional[np.ndarray] = None,
) -> Tuple[pd.Index, Optional[np.ndarray]]:
    """Return full future index for the group, and (if req_times) positions for them."""
    if req_times is not None:
        last_t = train_time_index[-1]
        max_t = pd.Index(req_times).max()

        if isinstance(train_time_index, pd.DatetimeIndex):
            raw_freq = train_time_index.freq or _infer_freq_from_index(train_time_index)
            offset = None
            if raw_freq is not None:
                try:
                    offset = to_offset(raw_freq)
                except (TypeError, ValueError):
                    offset = None
            if offset is None:
                inferred = pd.infer_freq(train_time_index)
                try:
                    offset = to_offset(inferred)
                except (TypeError, ValueError):
                    offset = None
            if offset is None:
                if len(train_time_index) >= 2:
                    step = train_time_index[-1] - train_time_index[-2]
                    try:
                        offset = to_offset(step)
                    except (TypeError, ValueError):
                        offset = None
            if offset is None:
                return pd.Index([]), np.array([], dtype=int)
            rng = pd.date_range(
                start=last_t + offset,
                end=max_t,
                freq=offset,
                tz=train_time_index.tz,
            )
            H = len(rng)
            full_future = pd.date_range(
                start=last_t + offset,
                periods=H if H > 0 else 0,
                freq=offset,
                tz=train_time_index.tz,
            )
        elif isinstance(train_time_index, pd.PeriodIndex):
            freq = train_time_index.freq
            rng = pd.period_range(start=last_t + 1, end=max_t, freq=freq)
            H = len(rng)
            full_future = pd.period_range(
                start=last_t + 1, periods=H if H > 0 else 0, freq=freq
            )
        elif is_integer_dtype(train_time_index.dtype):
            H = int(max_t - last_t)
            full_future = pd.RangeIndex(last_t + 1, last_t + 1 + max(H, 0))
        else:
            req_times_sorted = pd.Index(req_times).sort_values()
            H = len(req_times_sorted)
            full_future = pd.RangeIndex(1, H + 1)

        if H == 0:
            return pd.Index([]), np.array([], dtype=int)

        if isinstance(full_future, pd.RangeIndex) and not np.issubdtype(
            train_time_index.dtype, np.integer
        ):
            req_sorted = pd.Index(req_times).sort_values()
            pos_map = {req_sorted[i]: i + 1 for i in range(len(req_sorted))}
            steps = np.asarray([pos_map[t] for t in req_times], dtype=int)
        else:
            pos = pd.Index(full_future).get_indexer(pd.Index(req_times))
            if np.any(pos < 0):
                bad = list(pd.Index(req_times)[pos < 0][:3])
                raise ValueError(
                    "Requested times are not aligned with training frequency for a group. "
                    f"Examples: {bad}"
                )
            steps = pos.astype(int) + 1  # 1-based

        return full_future, steps

    # relative steps path
    if rel_steps is None or len(rel_steps) == 0:
        raise ValueError("Either req_times or rel_steps must be provided.")
    H = int(np.max(rel_steps))
    full_future, _ = _future_index_like(train_time_index, H)
    return pd.Index(full_future), None


def _union_indices(indices: List[pd.Index]) -> pd.Index:
    """Safe union for both Index and MultiIndex without relying on union_many."""
    if not indices:
        return pd.Index([])
    u = indices[0]
    for ix in indices[1:]:
        u = u.union(ix)
    return u


# ---------------------------------------------------------------------------
# The global forecaster
# ---------------------------------------------------------------------------


class ReductionForecaster(BaseForecaster):
    """Global hybrid reduction forecaster: K-step direct + recursive continuation.

    Trains **steps_ahead = K** separate direct models for horizons 1..K on pooled
    (possibly hierarchical) data, using a lag window from ``y`` (and optional
    *concurrent* exogenous ``X`` at each target timestamp). For each series id,
    requested predictions beyond K steps are produced recursively using the
    1-step model.

    This class works with **either** a single time series (simple time index) **or**
    MultiIndex/Hierarchical data (id levels + time).
    """

    _tags = {
        # accept single series AND hierarchical / multiindex series
        "y_inner_mtype": ["pd.Series", "pd-multiindex", "pd_multiindex_hier"],
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        # univariate target
        "scitype:y": "univariate",
        # exogenous supported
        "capability:exogenous": True,
        # does not require fh in fit
        "requires-fh-in-fit": False,
        # enforce same index between X and y
        "X-y-must-have-same-index": True,
        # index type unrestricted
        "enforce_index_type": None,
        # missing values: we don't guarantee generic handling (y can be imputed)
        "capability:missing_values": False,
        # strictly oos steps
        "capability:insample": False,
        # no probabilistic output in this implementation
        "capability:pred_int": False,
        # soft dependency on scikit-learn
        "python_dependencies": "scikit-learn",
    }

    def __init__(
        self,
        estimator: RegressorMixin,
        window_length: int = 10,
        steps_ahead: int = 1,
        normalization_strategy: Optional[
            Union[str, Callable[[np.ndarray], Tuple[Callable, Callable]]]
        ] = None,
        x_mode: str = "auto",
        impute_missing: Optional[str] = "bfill",
    ):
        # hyper-params
        self.estimator = estimator
        self.window_length = int(window_length)
        self.steps_ahead = int(steps_ahead)
        self.normalization_strategy = normalization_strategy
        self.x_mode = x_mode
        self.impute_missing = impute_missing

        super().__init__()

        if self.steps_ahead < 1:
            raise ValueError("steps_ahead must be a positive integer.")

        # learned attributes
        self._dir_estimators_: Optional[List[RegressorMixin]] = None
        self._estimator_: Optional[RegressorMixin] = None  # 1-step model shortcut
        self._x_used_: bool = False
        self._x_columns_: Optional[List[str]] = None

        # per-group rolling state
        self._last_windows_: Optional[Dict[Tuple, np.ndarray]] = (
            None  # ids -> window (old..new)
        )
        self._train_time_index_: Optional[Dict[Tuple, pd.Index]] = (
            None  # ids -> time index
        )
        self._ids_: Optional[List[Tuple]] = None  # list of ids tuples in fit order

        # index naming
        self._id_names_: Optional[List[str]] = None
        self._time_name_: Optional[str] = None
        self._was_single_series_: bool = False
        self._single_id_value_: str = "__singleton__"
        self._single_id_name_: str = "id"

        # for update/refit bookkeeping
        self._y_train_: Optional[pd.Series] = None
        self._X_train_: Optional[pd.DataFrame] = None
        self._norm_strategy_: Optional[
            Callable[[np.ndarray], Tuple[Callable, Callable]]
        ] = None
        self._y_name_: Optional[str] = None
        self._y_is_dataframe_: bool = False
        self._y_column_name_: Optional[str] = None

    # -------------------- fit --------------------
    def _fit(
        self, y: pd.Series, X: Optional[pd.DataFrame], fh: Optional[ForecastingHorizon]
    ):
        """Fit the global forecaster to (possibly hierarchical or single) training data."""
        _check_regressor(self.estimator)

        self._y_is_dataframe_ = isinstance(y, pd.DataFrame)
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(
                    "ReductionForecaster supports univariate targets only."
                )
            col_name = y.columns[0]
            y = y.iloc[:, 0].copy()
            y.name = col_name
            self._y_column_name_ = col_name
        else:
            self._y_column_name_ = y.name

        # detect single series and coerce to MultiIndex internally
        if isinstance(y.index, pd.MultiIndex) and y.index.nlevels >= 2:
            self._was_single_series_ = False
            y_mi = y.copy()
            if X is not None:
                if isinstance(X.index, pd.MultiIndex):
                    X_mi = X.copy()
                else:
                    raise TypeError(
                        "X must have a MultiIndex to match y's MultiIndex in fit."
                    )
            else:
                X_mi = None
        else:
            # single series -> wrap to MultiIndex with one id level
            self._was_single_series_ = True
            time_name = y.index.name if y.index.name is not None else "time"
            id_name = self._single_id_name_
            id_val = self._single_id_value_
            y_mi = y.copy()
            y_mi.index = pd.MultiIndex.from_arrays(
                [[id_val] * len(y_mi), y_mi.index], names=[id_name, time_name]
            )
            if X is not None:
                if isinstance(X.index, pd.MultiIndex):
                    raise TypeError(
                        "For single-series fit, X should have a simple time index."
                    )
                X_mi = X.copy()
                X_mi.index = pd.MultiIndex.from_arrays(
                    [[id_val] * len(X_mi), X_mi.index], names=[id_name, time_name]
                )
            else:
                X_mi = None

        y = y_mi
        X = X_mi

        # store names
        self._id_names_ = list(y.index.names[:-1])
        self._time_name_ = y.index.names[-1]
        self._y_name_ = y.name

        # basic imputation on y
        if self.impute_missing == "ffill":
            y = y.ffill()
        elif self.impute_missing == "bfill":
            y = y.bfill()
        elif self.impute_missing is not None:
            raise ValueError("impute_missing must be 'ffill', 'bfill', or None.")

        # resolve x_mode
        x_mode = self.x_mode
        if x_mode == "auto":
            x_mode = "concurrent" if X is not None else "none"

        # resolve normalization strategy (allow factory or strategy)
        self._norm_strategy_ = _resolve_normalization_strategy(
            self.normalization_strategy
        )

        # fit K direct heads using pooled data
        dir_estimators: List[RegressorMixin] = []
        for h in range(1, self.steps_ahead + 1):
            Xt_h_all, yt_h_all = _build_supervised_table_global(
                y=y,
                X=X,
                window_length=self.window_length,
                steps_ahead=h,
                x_mode=x_mode,
            )
            # remember X columns (consistency check at predict)
            if self._x_columns_ is None and X is not None and X.shape[1] > 0:
                self._x_columns_ = list(X.columns)

            # row-wise normalization on lags & target
            Xt_h_all_n, yt_h_all_n = _normalize_supervised_rowwise(
                Xt_h_all, yt_h_all, self.window_length, self._norm_strategy_
            )

            est_h = clone(self.estimator)
            est_h.fit(Xt_h_all_n.values, yt_h_all_n.values)
            dir_estimators.append(est_h)

        # learned state
        self._dir_estimators_ = dir_estimators
        self._estimator_ = dir_estimators[0]
        self._x_used_ = (x_mode == "concurrent") and (X is not None)

        # store per-group last window and time indices
        last_windows = {}
        time_idx_map = {}
        ids_list = []
        for ids, y_flat in _iter_series_groups(y):
            ids_list.append(ids)
            if not y_flat.index.is_monotonic_increasing:
                y_flat = y_flat.sort_index()
            time_idx_map[ids] = y_flat.index
            last = y_flat.iloc[-self.window_length :].to_numpy()
            if len(last) != self.window_length:
                raise ValueError(
                    f"Group {ids}: not enough observations for last window. "
                    f"Need window_length={self.window_length}, got {len(y_flat)}."
                )
            last_windows[ids] = last.astype(float).reshape(-1)

        self._last_windows_ = last_windows
        self._train_time_index_ = time_idx_map
        self._ids_ = ids_list

        # store training for potential refit in update
        self._y_train_ = y.copy()
        self._X_train_ = X.copy() if X is not None else None

        return self

    # -------------------- predict --------------------
    def _predict(
        self,
        fh: Union[ForecastingHorizon, Sequence, pd.Index, pd.MultiIndex],
        X: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Forecast pooled multi-series or a single series at future horizon."""
        if (
            self._estimator_ is None
            or self._last_windows_ is None
            or self._dir_estimators_ is None
            or self._train_time_index_ is None
            or self._ids_ is None
        ):
            raise RuntimeError("Call fit(...) before predict(...).")

        # determine fh mode
        mode_abs_multi = isinstance(fh, pd.MultiIndex)
        mode_abs_single = isinstance(fh, pd.Index) and not isinstance(fh, pd.MultiIndex)
        mode_rel = False

        req_steps_all: Optional[np.ndarray] = None
        if not (mode_abs_multi or mode_abs_single):
            # FH object or array-like of relative ints
            if isinstance(fh, ForecastingHorizon):
                rel = fh.to_relative(self.cutoff)
                req_steps_all = _as_positive_int_fh(np.asarray(rel, dtype=int))
            else:
                # try array-like relative ints
                arr = np.asarray(fh)
                req_steps_all = _as_positive_int_fh(arr)
            mode_rel = True

        if mode_abs_single and not self._was_single_series_:
            raise TypeError(
                "Absolute fh as a simple Index is only valid when the model was fit "
                "on a single series. For multi-series, pass a MultiIndex fh."
            )

        # prepare exogenous usage
        if self._x_used_:
            if X is None:
                raise ValueError(
                    "This model was fit with exogenous variables. "
                    "Provide X with rows for all required forecast timestamps."
                )
            if self._x_columns_ is not None:
                missing = [c for c in self._x_columns_ if c not in X.columns]
                if missing:
                    raise ValueError(
                        f"X is missing columns seen in training: {missing}"
                    )
            # shape validation
            if self._was_single_series_:
                if isinstance(X.index, pd.MultiIndex):
                    raise TypeError(
                        "For single-series prediction, X should have a simple time index."
                    )
            else:
                if not isinstance(X.index, pd.MultiIndex):
                    raise TypeError(
                        "For multi-series prediction, X must have a MultiIndex index."
                    )
            # order X columns to match training
            if self._x_columns_ is not None:
                X = X[self._x_columns_]

        out_series: List[pd.Series] = []

        K = self.steps_ahead

        for ids in self._ids_:
            time_idx_train = self._train_time_index_[ids]

            # determine requested times/steps for this ids
            if mode_abs_multi:
                try:
                    req_times = fh.xs(ids, level=list(range(fh.nlevels - 1)))
                    req_times = pd.Index(req_times)
                except Exception:
                    req_times = pd.Index([])
                full_future, steps_for_req = _steps_and_full_future_for_group(
                    time_idx_train, req_times=req_times
                )
                if len(full_future) == 0 and len(req_times) == 0:
                    continue
                H = len(full_future)
                pos_req = steps_for_req  # 1-based
            elif mode_abs_single:
                # single series: use the provided absolute time Index for the lone ids
                req_times = pd.Index(fh)
                full_future, steps_for_req = _steps_and_full_future_for_group(
                    time_idx_train, req_times=req_times
                )
                if len(full_future) == 0 and len(req_times) == 0:
                    continue
                H = len(full_future)
                pos_req = steps_for_req
            else:
                # relative steps (common to all ids)
                assert req_steps_all is not None
                H = int(np.max(req_steps_all))
                full_future, _ = _steps_and_full_future_for_group(
                    time_idx_train, rel_steps=req_steps_all
                )
                pos_req = req_steps_all

            # prepare exogenous block for this group's full future horizon
            X_block = None
            if self._x_used_:
                if self._was_single_series_:
                    X_needed = _select_future_rows(X, full_future)
                    X_block = X_needed.to_numpy()
                else:
                    group_future_index = _make_group_future_multiindex(
                        ids, full_future, self._id_names_, self._time_name_
                    )
                    X_needed = _select_future_rows(X, group_future_index)
                    X_block = X_needed.to_numpy()

            # predictions for steps 1..H
            preds = np.zeros(H, dtype=float)

            # direct part
            last_obs = self._last_windows_[ids].copy()  # chronological old..new
            for i in range(1, min(K, H) + 1):
                y_feats = last_obs[::-1]  # newest first to match training
                if self._norm_strategy_ is not None:
                    transform, inv = self._norm_strategy_(y_feats)
                    y_feats_n, _ = transform(y_feats, None)
                else:
                    y_feats_n = y_feats
                    inv = lambda v: float(v)

                if X_block is not None:
                    row = np.concatenate([y_feats_n, X_block[i - 1]])
                else:
                    row = y_feats_n

                yhat_n = float(
                    np.asarray(
                        self._dir_estimators_[i - 1].predict(row.reshape(1, -1))
                    ).ravel()[0]
                )
                yhat = inv(yhat_n)
                preds[i - 1] = yhat

            # rolling state after K direct preds
            last_roll = self._last_windows_[ids].copy()
            for i in range(1, min(K, H) + 1):
                last_roll = np.roll(last_roll, -1)
                last_roll[-1] = preds[i - 1]

            # recursive part
            for i in range(K + 1, H + 1):
                y_feats = last_roll[::-1]
                if self._norm_strategy_ is not None:
                    transform, inv = self._norm_strategy_(y_feats)
                    y_feats_n, _ = transform(y_feats, None)
                else:
                    y_feats_n = y_feats
                    inv = lambda v: float(v)

                if X_block is not None:
                    row = np.concatenate([y_feats_n, X_block[i - 1]])
                else:
                    row = y_feats_n

                yhat_n = float(
                    np.asarray(self._estimator_.predict(row.reshape(1, -1))).ravel()[0]
                )
                yhat = inv(yhat_n)
                preds[i - 1] = yhat

                # roll forward with *original-scale* prediction
                last_roll = np.roll(last_roll, -1)
                last_roll[-1] = yhat

            # subset to requested steps for this ids and append to output
            steps = np.asarray(pos_req, dtype=int)
            sel = preds[steps - 1]

            if mode_abs_multi:
                idx = _make_group_future_multiindex(
                    ids, full_future[steps - 1], self._id_names_, self._time_name_
                )
            elif mode_abs_single or (mode_rel and self._was_single_series_):
                idx = pd.Index(full_future[steps - 1], name=self._time_name_)
            else:
                idx = _make_group_future_multiindex(
                    ids, full_future[steps - 1], self._id_names_, self._time_name_
                )

            out_series.append(pd.Series(sel, index=idx, name=self._y_name_))

        if len(out_series) == 0:
            # No requested rows (e.g., fh had no times beyond training) -> empty series
            return pd.Series([], dtype=float, name=self._y_name_)

        # Assemble output
        y_pred = pd.concat(out_series)

        # preserve the order of the provided absolute fh if given
        if mode_abs_multi:
            y_pred = y_pred.reindex(fh)
        elif mode_abs_single:
            y_pred = y_pred.reindex(pd.Index(fh))
        else:
            y_pred = y_pred.sort_index()

        if self._y_is_dataframe_:
            col_name = self._y_column_name_ or self._y_name_ or "y"
            return y_pred.to_frame(name=col_name)

        return y_pred

    # -------------------- update --------------------
    def _update(
        self, y: pd.Series, X: Optional[pd.DataFrame] = None, update_params: bool = True
    ):
        """Update rolling windows; refit on appended data if `update_params=True`."""
        if (
            self._estimator_ is None
            or self._last_windows_ is None
            or self._dir_estimators_ is None
            or self._train_time_index_ is None
            or self._ids_ is None
        ):
            raise RuntimeError("Call fit(...) before update(...).")

        if y is None or len(y) == 0:
            return self

        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(
                    "ReductionForecaster update expects a single target column."
                )
            col_name = y.columns[0]
            y = y.iloc[:, 0].copy()
            y.name = col_name

        # Coerce y (and X) to the internal MultiIndex shape if needed
        if self._was_single_series_:
            time_name = self._time_name_ or "time"
            id_name = self._id_names_[0] if self._id_names_ else self._single_id_name_
            id_val = self._single_id_value_
            if not isinstance(y.index, pd.MultiIndex):
                y = y.copy()
                y.index = pd.MultiIndex.from_arrays(
                    [[id_val] * len(y), y.index], names=[id_name, time_name]
                )
            if X is not None and not isinstance(X.index, pd.MultiIndex):
                X = X.copy()
                X.index = pd.MultiIndex.from_arrays(
                    [[id_val] * len(X), X.index], names=[id_name, time_name]
                )

        # roll last windows for groups present in y
        for ids, y_flat in _iter_series_groups(y):
            new_vals = y_flat.to_numpy(dtype=float).reshape(-1)
            if len(new_vals) == 0:
                continue
            if len(new_vals) >= self.window_length:
                self._last_windows_[ids] = new_vals[-self.window_length :]
            else:
                rolled = np.roll(self._last_windows_[ids], -len(new_vals))
                rolled[-len(new_vals) :] = new_vals
                self._last_windows_[ids] = rolled
            # update stored time index for the group
            existing_index = self._train_time_index_[ids]
            new_index = y_flat.index
            if isinstance(existing_index, pd.DatetimeIndex):
                combined = existing_index.append(pd.DatetimeIndex(new_index))
            elif isinstance(existing_index, pd.PeriodIndex):
                combined = existing_index.append(
                    pd.PeriodIndex(new_index, freq=existing_index.freq)
                )
            else:
                combined = existing_index.append(pd.Index(new_index))

            if not combined.is_monotonic_increasing:
                combined = combined.sort_values()

            self._train_time_index_[ids] = combined

        if not update_params:
            return self

        # Refit from scratch on concatenated data (simple & robust)
        y_full = pd.concat([self._y_train_, y]).sort_index()
        X_full = None
        if self._X_train_ is not None or X is not None:
            if (self._X_train_ is not None) and (X is None):
                raise ValueError(
                    "This model was originally fit with X; update requires matching X."
                )
            X_full = pd.concat([self._X_train_, X]).sort_index()

        _check_regressor(self.estimator)

        # impute like in fit
        if self.impute_missing == "ffill":
            y_imp = y_full.ffill()
        elif self.impute_missing == "bfill":
            y_imp = y_full.bfill()
        else:
            y_imp = y_full.copy()

        x_mode = self.x_mode
        if x_mode == "auto":
            x_mode = "concurrent" if X_full is not None else "none"

        dir_estimators: List[RegressorMixin] = []
        for h in range(1, self.steps_ahead + 1):
            Xt_h_all, yt_h_all = _build_supervised_table_global(
                y=y_imp,
                X=X_full,
                window_length=self.window_length,
                steps_ahead=h,
                x_mode=x_mode,
            )
            Xt_h_all_n, yt_h_all_n = _normalize_supervised_rowwise(
                Xt_h_all, yt_h_all, self.window_length, self._norm_strategy_
            )
            est_h = clone(self.estimator)
            est_h.fit(Xt_h_all_n.values, yt_h_all_n.values)
            dir_estimators.append(est_h)

        # update learned state
        self._dir_estimators_ = dir_estimators
        self._estimator_ = dir_estimators[0]
        self._x_used_ = (x_mode == "concurrent") and (X_full is not None)
        self._x_columns_ = (
            list(X_full.columns) if (X_full is not None) else self._x_columns_
        )

        # refresh per-group windows and time index maps from y_imp
        last_windows = {}
        time_idx_map = {}
        ids_list = []
        for ids, y_flat in _iter_series_groups(y_imp):
            ids_list.append(ids)
            if not y_flat.index.is_monotonic_increasing:
                y_flat = y_flat.sort_index()
            time_idx_map[ids] = y_flat.index
            last = y_flat.iloc[-self.window_length :].to_numpy()
            last_windows[ids] = last.astype(float).reshape(-1)
        self._last_windows_ = last_windows
        self._train_time_index_ = time_idx_map
        self._ids_ = ids_list

        # store full data for potential next update
        self._y_train_ = y_full.copy()
        self._X_train_ = X_full.copy() if X_full is not None else None

        return self

    # -------------------- fitted params --------------------
    def _get_fitted_params(self):
        """Expose fitted parameters and learned state."""
        return {
            "x_used": self._x_used_,
            "x_columns": self._x_columns_,
            "direct_estimators": self._dir_estimators_,
            "one_step_estimator": self._estimator_,
            "last_windows": {
                k: v.copy() for k, v in (self._last_windows_ or {}).items()
            },
            "id_names": self._id_names_,
            "time_name": self._time_name_,
            "was_single_series": self._was_single_series_,
            "y_was_dataframe": self._y_is_dataframe_,
        }

    # -------------------- test params --------------------
    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        """Return parameter settings for the estimator tests."""
        from sklearn.linear_model import LinearRegression, Ridge

        if parameter_set == "fast":
            return {
                "estimator": LinearRegression(),
                "window_length": 4,
                "steps_ahead": 2,
                "normalization_strategy": mean_window_normalizer,  # factory form
            }

        return [
            {
                "estimator": LinearRegression(),
                "window_length": 5,
                "steps_ahead": 1,
                "normalization_strategy": mean_window_normalizer(),  # strategy form
            },
            {
                "estimator": Ridge(alpha=0.1),
                "window_length": 3,
                "steps_ahead": 3,
                "normalization_strategy": None,
            },
        ]


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def make_reduction(
    estimator: RegressorMixin,
    strategy: str = "recursive",
    window_length: int = 10,
    steps_ahead: Optional[int] = None,
    normalization_strategy: Optional[
        Union[str, Callable[[np.ndarray], Tuple[Callable, Callable]]]
    ] = None,
    x_mode: str = "auto",
    impute_missing: Optional[str] = "bfill",
) -> ReductionForecaster:
    """
    Construct a ReductionForecaster.

    In this unified design:
    - If ``strategy='recursive'`` and ``steps_ahead is None``, you'll get K=1.
    - If ``strategy='direct'`` and you pass ``steps_ahead=K``, you'll get K direct heads
      for 1..K and recursive continuation beyond K.
    - Any other combination behaves the same as setting K=max(1, steps_ahead).
    """
    strategy = (strategy or "recursive").lower()
    if strategy not in ("recursive", "direct"):
        raise ValueError("strategy must be 'recursive' or 'direct'.")

    if steps_ahead is None:
        K = 1
    else:
        K = int(steps_ahead)
        if K < 1:
            raise ValueError("steps_ahead must be a positive integer.")

    return ReductionForecaster(
        estimator=estimator,
        window_length=window_length,
        steps_ahead=K,
        normalization_strategy=normalization_strategy,
        x_mode=x_mode,
        impute_missing=impute_missing,
    )


# ---------------------------------------------------------------------------
# Minimal smoke test (optional)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)

    # -------- Single series example --------
    n = 50
    t = pd.date_range("2024-01-01", periods=n, freq="D")
    y_single = pd.Series(
        np.sin(np.linspace(0, 4, n)) + 0.1 * rng.standard_normal(n),
        index=t,
        name="y",
    )
    X_single = pd.DataFrame({"cos": np.cos(np.linspace(0, 4, n))}, index=t)

    f_single = make_reduction(
        LinearRegression(),
        strategy="direct",
        window_length=7,
        steps_ahead=3,
        normalization_strategy=mean_window_normalizer,  # factory OR mean_window_normalizer()
    )
    f_single.fit(y_single, X=X_single)

    H = 5
    future_times = pd.date_range(t[-1] + pd.Timedelta(days=1), periods=H, freq="D")
    Xf_single = pd.DataFrame(
        {"cos": np.cos(np.linspace(4, 4 + 0.05 * H, H))}, index=future_times
    )
    print("Single-series forecast:")
    print(f_single.predict(fh=future_times, X=Xf_single))

    # -------- Multi-series example --------
    ids = ["A", "B"]
    ys = []
    Xs = []
    for i, s in enumerate(ids):
        y = pd.Series(
            np.sin(np.linspace(0, 4, n)) + 0.1 * rng.standard_normal(n) + i,
            index=t,
            name="y",
        )
        y.index = pd.MultiIndex.from_product([[s], y.index], names=["id", "time"])
        ys.append(y)

        X = pd.DataFrame({"cos": np.cos(np.linspace(0, 4, n))}, index=t)
        X.index = pd.MultiIndex.from_product([[s], X.index], names=["id", "time"])
        Xs.append(X)

    y_all = pd.concat(ys)
    X_all = pd.concat(Xs)

    f_multi = make_reduction(
        LinearRegression(),
        strategy="direct",
        window_length=7,
        steps_ahead=3,
        normalization_strategy=mean_window_normalizer,
    )
    f_multi.fit(y_all, X=X_all)

    future_times = pd.date_range(t[-1] + pd.Timedelta(days=1), periods=H, freq="D")
    fh_abs = pd.MultiIndex.from_product([ids, future_times], names=["id", "time"])
    Xf = []
    for s in ids:
        Xs_f = pd.DataFrame(
            {"cos": np.cos(np.linspace(4, 4 + 0.05 * H, H))}, index=future_times
        )
        Xs_f.index = pd.MultiIndex.from_product([[s], Xs_f.index], names=["id", "time"])
        Xf.append(Xs_f)
    Xf = pd.concat(Xf)

    print("\nMulti-series forecast:")
    print(f_multi.predict(fh=fh_abs, X=Xf))
