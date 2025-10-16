#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# copyright: (c) 2025, authored for the requesting user
# license: BSD-3-Clause-compatible grant for this file by the author
"""
Hybrid reduction forecaster for sktime: K-step direct + recursive continuation.

This forecaster inherits from ``sktime.forecasting.base.BaseForecaster`` and can
train *K* inner direct models to forecast 1..K steps ahead from a shared lagged
feature window of ``y`` (and optional *concurrent* exogenous ``X``). When asked
to predict beyond K steps, it *recursively* rolls a 1-step model forward using
its own past predictions. Setting ``steps_ahead=1`` recovers pure recursive
reduction. In that sense, **DirectReduction is a subcase of RecursiveReduction
with steps_ahead=1**; larger ``steps_ahead`` simply adds direct heads for the
first K steps before recursion continues them.

New: per-window normalization via a lightweight strategy callback
-----------------------------------------------------------------
Pass ``normalization_strategy`` as a **callable** that receives the y-lag window
vector (1D ndarray, ordered as features are fed to the model: [y_t, y_{t-1}, ...])
and returns a pair of functions ``(transform, inverse_transform)``. These functions
must each accept and return a 1D ndarray. The same transform is applied to the
lag window **and** the scalar target for that row; predictions are immediately
inverse-transformed back to the original scale. Exogenous ``X`` is never normalized.

Example strategy (provided below): :func:`meanvar_window_normalizer`, which centers
and scales by the window's mean and standard deviation.

Highlights
----------
- Works with any scikit-learn style regressor (fit/predict).
- Univariate ``y``; optional *concurrent* exogenous ``X`` (values at the *target*
  timestamps). If used in ``fit``, you must provide future ``X`` rows in ``predict``.
- Handles arbitrary forecasting horizons (not necessarily consecutive). Internally
  computes predictions for steps 1..H where H=max requested step, then subselects.

Notes
-----
- This is a brand-new implementation authored from scratch and not copied from
  sktime or other libraries. It follows the sktime extension template.
- Scope is intentionally minimal: single series (no panel/global), point forecasts.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import clone, RegressorMixin
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


__all__ = [
    "ReductionForecaster",
    "make_reduction",
    "meanvar_window_normalizer",
]


# ---------------------------------------------------------------------------
# utils (pure-python; no sktime private imports)
# ---------------------------------------------------------------------------

# Type alias for normalization strategy:
# given a 1D window vector (lags), return (transform, inverse_transform) pair
# both functions operate on 1D ndarrays and must be shape-preserving.
NormStrategy = Optional[
    Callable[
        [np.ndarray],
        Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]],
    ]
]


def _ensure_series(y: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """Coerce y to a univariate Series (first column if DataFrame)."""
    if isinstance(y, pd.Series):
        return y
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 0:
            raise ValueError("y DataFrame has no columns.")
        return y.iloc[:, 0]
    raise TypeError("y must be a pandas Series or DataFrame.")


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
        freq = idx.freq or _infer_freq_from_index(idx)
        if freq is not None:
            start = idx[-1] + freq
            return (
                pd.date_range(start=start, periods=horizon, freq=freq, tz=idx.tz),
                True,
            )
        return pd.RangeIndex(1, horizon + 1), False

    if isinstance(idx, pd.PeriodIndex):
        freq = idx.freq
        if freq is not None:
            start = idx[-1] + 1
            return pd.period_range(start=start, periods=horizon, freq=freq), True
        return pd.RangeIndex(1, horizon + 1), False

    if isinstance(
        idx, (pd.RangeIndex, pd.Int64Index, pd.UInt64Index, pd.Index)
    ) and np.issubdtype(idx.dtype, np.integer):
        start = idx[-1] + 1
        return pd.RangeIndex(start, start + horizon), True

    return pd.RangeIndex(1, horizon + 1), False


def _build_supervised_table(
    y: pd.Series,
    X: Optional[pd.DataFrame],
    window_length: int,
    steps_ahead: int,
    x_mode: str,
    normalization_strategy: NormStrategy = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Turn (y, X) into (Xt, yt) for supervised learning for a single horizon.

    - Features: y lags [t, t-1, ..., t - window_length + 1]  (most recent first)
    - Target:   y[t + steps_ahead]
    - Exogenous concurrent: X at time t + steps_ahead (if used)
    - Normalization: if `normalization_strategy` is provided, for each row:
        * obtain (transform, inverse_transform) = normalization_strategy(y_lag_vector)
        * replace lag features by transform(y_lag_vector)
        * replace scalar target by transform([target])[0]
      Note: X is *not* normalized.
    """
    if window_length < 1:
        raise ValueError("window_length must be >= 1.")
    if not isinstance(steps_ahead, int) or steps_ahead < 1:
        raise ValueError("steps_ahead must be a positive integer.")
    if x_mode not in ("none", "concurrent", "auto"):
        raise ValueError("x_mode must be one of {'none', 'concurrent', 'auto'}.")

    use_X = (X is not None) and (x_mode in ("concurrent", "auto"))
    if use_X:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame when provided.")
        if not X.index.is_monotonic_increasing:
            X = X.sort_index()
        if not y.index.is_monotonic_increasing:
            y = y.sort_index()

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

    x_cols = []
    if use_X:
        x_cols = list(X.columns)

    for t in range(window_length - 1, max_anchor + 1):
        # y lags vector in the same order as features will be fed: [y_t, y_{t-1}, ...]
        lag_block = values[t : t - window_length : -1]
        if lag_block.shape[0] != window_length:
            # safety for very early slices (rarely hit)
            lag_block = np.asarray(
                [values[t - i] for i in range(window_length)], dtype=float
            )
        y_feats = lag_block.astype(float).copy()

        # target scalar at t + h
        target_time = y.index[t + steps_ahead]
        y_target = float(values[t + steps_ahead])

        # apply per-row normalization if requested
        if normalization_strategy is not None:
            tr, _inv = normalization_strategy(y_feats.copy())
            y_feats = np.asarray(tr(y_feats), dtype=float)
            y_target = float(np.asarray(tr(np.array([y_target], dtype=float)))[0])

        # build row dict
        row = {f"y_lag_{i+1}": y_feats[i] for i in range(window_length)}

        # append X (concurrent at target_time) without normalization
        if use_X:
            if target_time not in X.index:
                for c in x_cols:
                    row[f"X_{c}"] = np.nan
            else:
                xrow = X.loc[target_time]
                if isinstance(xrow, pd.DataFrame):
                    xrow = xrow.iloc[0]
                for c in x_cols:
                    row[f"X_{c}"] = xrow[c]

        rows.append(row)
        targets.append(y_target)
        t_index.append(target_time)

    Xt = pd.DataFrame(rows, index=pd.Index(t_index, name=y.index.name))
    yt = pd.Series(targets, index=Xt.index, name=y.name)
    return Xt, yt


def _select_future_rows(X_future: pd.DataFrame, idx: pd.Index) -> pd.DataFrame:
    """Select rows of X_future at index `idx`, raising error if any missing."""
    idx = pd.Index(idx)
    missing = idx.difference(X_future.index)
    if len(missing) > 0:
        sample = list(missing[:3])
        raise ValueError(
            "Missing required rows in X for forecast timestamps. "
            f"Examples: {sample} (total missing: {len(missing)})."
        )
    return X_future.loc[idx]


def _fh_to_absolute_index(
    fh_like: Union[ForecastingHorizon, Sequence, pd.Index],
    cutoff,
    y_index: pd.Index,
    steps: Optional[Sequence[int]] = None,
    H: Optional[int] = None,
) -> pd.Index:
    """
    Robustly coerce a forecasting horizon (absolute or relative) to a pandas Index.

    Tries multiple sktime FH APIs across versions, then falls back to constructing
    a "future index like y_index".
    """
    # If it's already a pandas Index, return it
    if isinstance(fh_like, pd.Index):
        return fh_like

    # If it's not an FH, try to coerce directly
    if not isinstance(fh_like, ForecastingHorizon):
        try:
            return pd.Index(fh_like)
        except Exception:
            pass  # fall through to robust fallback

    # From here treat as ForecastingHorizon
    fh_obj = (
        fh_like
        if isinstance(fh_like, ForecastingHorizon)
        else ForecastingHorizon(fh_like)
    )

    # 1) Preferred: to_absolute_index(cutoff)
    m = getattr(fh_obj, "to_absolute_index", None)
    if callable(m):
        try:
            return m(cutoff)
        except Exception:
            pass

    # 2) Some versions: to_pandas_index() if already absolute
    m = getattr(fh_obj, "to_pandas_index", None)
    if callable(m):
        try:
            idx = m()
            if isinstance(idx, pd.Index):
                return idx
        except Exception:
            pass

    # 3) Try going to absolute first, then 1) and 2)
    try:
        abs_fh = fh_obj.to_absolute(cutoff)
        m = getattr(abs_fh, "to_absolute_index", None)
        if callable(m):
            try:
                return m(cutoff)
            except Exception:
                pass
        m = getattr(abs_fh, "to_pandas_index", None)
        if callable(m):
            try:
                idx = m()
                if isinstance(idx, pd.Index):
                    return idx
            except Exception:
                pass
        # If abs_fh itself is index-like
        if not isinstance(abs_fh, ForecastingHorizon):
            try:
                return pd.Index(abs_fh)
            except Exception:
                pass
    except Exception:
        pass

    # 4) Last resort: synthesize using y_index's cadence
    if steps is not None:
        steps = np.asarray(steps, dtype=int).reshape(-1)
        H_ = int(np.max(steps))
    else:
        H_ = int(H if H is not None else len(fh_obj))
        steps = np.arange(1, H_ + 1, dtype=int)

    full_future, _ = _future_index_like(y_index, H_)
    return pd.Index([full_future[h - 1] for h in steps])


# ---------------------------------------------------------------------------
# The forecaster
# ---------------------------------------------------------------------------


class ReductionForecaster(BaseForecaster):
    """Hybrid reduction forecaster: K-step direct + recursive continuation.

    Trains **steps_ahead = K** separate direct models for horizons 1..K using a
    lag window from ``y`` (and optional *concurrent* exogenous ``X`` at each
    target timestamp). For a requested forecast horizon H:

    - For steps 1..min(K, H): use the corresponding direct model h to predict y[h]
      from the *same observed* lag window (no predicted values fed back here).
    - If H > K: continue with **recursive** one-step predictions, starting from the
      observed lag window rolled forward by the K *direct* predictions, and use the
      trained 1-step model repeatedly.

    Normalization strategy
    ----------------------
    The optional ``normalization_strategy`` is a callable that receives the
    **current y-lag window** (1D ndarray in feature order: [y_t, y_{t-1}, ...]) and
    returns a pair of functions ``(transform, inverse_transform)``, both operating on
    1D ndarrays. In training, for each supervised row, we fit this per-window
    normalizer on the y-lag vector, transform the lags **and** the scalar target,
    train models in the normalized space, and in prediction we inverse-transform each
    predicted scalar immediately back to the original y-scale.

    Parameters
    ----------
    estimator : sklearn-style regressor
        Any object with `fit(X, y)` and `predict(X)` methods.
    window_length : int, default=10
        Number of past observations to use as lags.
    steps_ahead : int, default=1
        Number of direct heads (K). K=1 recovers pure recursive reduction.
    x_mode : {"auto","none","concurrent"}, default="auto"
        - "none": ignore X even if provided
        - "concurrent": use X at the **target** timestamps in `fit`, and the
          future timestamps in `predict`
        - "auto": behaves like "concurrent" if X is provided else "none"
    impute_missing : {"ffill","bfill",None}, default="bfill"
        Optional imputation applied to `y` **before** windowing.
        If None, NaNs are left as-is (your estimator must handle them).
    normalization_strategy : callable or None, default=None
        Function ``f(window_1d) -> (transform, inverse_transform)``. If None, no
        normalization is applied.

    Notes
    -----
    - Univariate series only (single variable).
    - If X is used in fit, you must pass future X at *all* required forecast
      timestamps to `predict`.
    """

    # -------------------- sktime estimator tags --------------------
    _tags = {
        # inner mtypes
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        # univariate only
        "scitype:y": "univariate",
        # exogenous supported
        "capability:exogenous": True,
        # does not require fh in fit
        "requires-fh-in-fit": False,
        # enforce same index between X and y
        "X-y-must-have-same-index": True,
        # index type unrestricted
        "enforce_index_type": None,
        # we don't guarantee missing handling in general (y can be imputed)
        "capability:missing_values": False,
        # only strictly out-of-sample fh supported (positive relative steps)
        "capability:insample": False,
        # no probabilistic output
        "capability:pred_int": False,
        # soft dependency on sklearn (sktime tag uses the import name)
        "python_dependencies": "scikit-learn",
    }

    # -------------------- constructor signature --------------------
    def __init__(
        self,
        estimator: RegressorMixin,
        window_length: int = 10,
        steps_ahead: int = 1,
        x_mode: str = "auto",
        impute_missing: Optional[str] = "bfill",
        normalization_strategy: NormStrategy = None,
    ):
        # components / hyper-params
        self.estimator = estimator
        self.window_length = int(window_length)
        self.steps_ahead = int(steps_ahead)
        self.x_mode = x_mode
        self.impute_missing = impute_missing
        self.normalization_strategy = normalization_strategy

        super().__init__()

        if self.steps_ahead < 1:
            raise ValueError("steps_ahead must be a positive integer.")
        if (self.normalization_strategy is not None) and (
            not callable(self.normalization_strategy)
        ):
            raise TypeError("normalization_strategy must be a callable or None.")

        # learned attributes (set in _fit)
        self._dir_estimators_: Optional[List[RegressorMixin]] = None
        self._estimator_: Optional[RegressorMixin] = None  # alias: 1-step model
        self._x_used_: bool = False
        self._x_columns_: Optional[List[str]] = None
        self._last_window_: Optional[np.ndarray] = None
        self._y_train_index_: Optional[pd.Index] = None
        self._y_name_: Optional[str] = None
        self._y_train_: Optional[pd.Series] = None
        self._X_train_: Optional[pd.DataFrame] = None

    # -------------------- fit logic --------------------
    def _fit(
        self, y: pd.Series, X: Optional[pd.DataFrame], fh: Optional[ForecastingHorizon]
    ):
        """Fit forecaster to training data (private core, called by BaseForecaster)."""
        _check_regressor(self.estimator)

        y = _ensure_series(y).copy()

        # basic imputation on y
        if self.impute_missing in ("ffill", "bfill"):
            y = y.fillna(method=self.impute_missing)
        elif self.impute_missing is not None:
            raise ValueError("impute_missing must be 'ffill', 'bfill', or None.")

        # resolve x_mode
        x_mode = self.x_mode
        if x_mode == "auto":
            x_mode = "concurrent" if X is not None else "none"

        # fit K direct heads (1..steps_ahead)
        dir_estimators: List[RegressorMixin] = []
        for h in range(1, self.steps_ahead + 1):
            Xt_h, yt_h = _build_supervised_table(
                y=y,
                X=X,
                window_length=self.window_length,
                steps_ahead=h,
                x_mode=x_mode,
                normalization_strategy=self.normalization_strategy,
            )
            est_h = clone(self.estimator)
            est_h.fit(Xt_h.values, yt_h.values)
            dir_estimators.append(est_h)

        # learned state
        self._dir_estimators_ = dir_estimators
        self._estimator_ = dir_estimators[0]  # 1-step model
        self._x_used_ = (x_mode == "concurrent") and (X is not None)
        self._x_columns_ = list(X.columns) if (X is not None) else None
        self._y_train_index_ = y.index
        self._y_name_ = y.name

        # bootstrap last window for recursion (stored oldest->newest, as y.iloc preserves)
        last = y.iloc[-self.window_length :].to_numpy()
        if len(last) != self.window_length:
            raise ValueError(
                "Not enough observations to form last window. "
                f"Need window_length={self.window_length}, got {len(y)}."
            )
        self._last_window_ = last.astype(float).reshape(-1)

        # store training series for optional updates
        self._y_train_ = y.copy()
        self._X_train_ = X.copy() if X is not None else None

        return self

    # -------------------- predict logic --------------------
    def _predict(
        self, fh: ForecastingHorizon, X: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """Forecast time series at future horizon (private core, called by BaseForecaster)."""
        if (
            self._estimator_ is None
            or self._last_window_ is None
            or self._dir_estimators_ is None
        ):
            raise RuntimeError("Call fit(...) before predict(...).")

        # relative steps (strictly positive due to tag capability:insample=False)
        rel = fh.to_relative(self.cutoff)
        rel_steps = np.asarray(rel, dtype=int).reshape(-1)
        pos_steps = _as_positive_int_fh(rel_steps)
        H = int(pos_steps.max())

        # build absolute indexes robustly (supports various sktime versions)
        all_rel = ForecastingHorizon(np.arange(1, H + 1, dtype=int), is_relative=True)
        abs_all_like = all_rel.to_absolute(self.cutoff)
        abs_all_idx = _fh_to_absolute_index(
            abs_all_like, cutoff=self.cutoff, y_index=self._y_train_index_, H=H
        )

        abs_req_like = fh.to_absolute(self.cutoff)
        abs_req_idx = _fh_to_absolute_index(
            abs_req_like,
            cutoff=self.cutoff,
            y_index=self._y_train_index_,
            steps=rel_steps,
            H=H,
        )

        # if X was used in fit, we need concurrent X for *all* 1..H steps
        if self._x_used_:
            if X is None:
                raise ValueError(
                    "This model was fit with exogenous variables. "
                    "Provide X with rows for all required forecast timestamps."
                )
            if not isinstance(X, pd.DataFrame):
                raise TypeError(
                    "X must be a pandas DataFrame when provided to predict."
                )
            if not X.index.is_monotonic_increasing:
                X = X.sort_index()

            X_all = _select_future_rows(X, abs_all_idx)
            if self._x_columns_ is not None:
                missing_cols = [c for c in self._x_columns_ if c not in X_all.columns]
                if missing_cols:
                    raise ValueError(
                        f"X is missing columns seen in training: {missing_cols}"
                    )
                X_all = X_all[self._x_columns_]
            X_block = X_all.to_numpy()
        else:
            X_block = None

        preds = np.zeros(H, dtype=float)

        # ----- Direct part (1..min(K, H)) using *observed* window only -----
        K = min(self.steps_ahead, H)
        last_obs = self._last_window_.copy()  # oldest -> newest
        for i in range(1, K + 1):
            # features in model order: most recent first
            y_feats = last_obs[::-1]  # y_lag_1 := most recent true observation

            # per-step normalization (fit on current window)
            if self.normalization_strategy is not None:
                tr, inv = self.normalization_strategy(y_feats.copy())
                y_feats_n = np.asarray(tr(y_feats), dtype=float)
            else:
                # identity mapping
                y_feats_n = y_feats
                inv = lambda a: a  # noqa: E731

            if X_block is not None:
                row = np.concatenate([y_feats_n, X_block[i - 1]])
            else:
                row = y_feats_n

            yhat_norm = float(
                np.asarray(
                    self._dir_estimators_[i - 1].predict(row.reshape(1, -1))
                ).ravel()[0]
            )
            # map back to original scale
            yhat = float(np.asarray(inv(np.array([yhat_norm], dtype=float)))[0])
            preds[i - 1] = yhat

        # ----- Prepare rolling state after K direct steps -----
        last_roll = self._last_window_.copy()
        for i in range(1, K + 1):
            last_roll = np.roll(last_roll, -1)
            last_roll[-1] = preds[i - 1]

        # ----- Recursive continuation (K+1..H) using 1-step model -----
        for i in range(K + 1, H + 1):
            y_feats = last_roll[::-1]

            if self.normalization_strategy is not None:
                tr, inv = self.normalization_strategy(y_feats.copy())
                y_feats_n = np.asarray(tr(y_feats), dtype=float)
            else:
                y_feats_n = y_feats
                inv = lambda a: a  # noqa: E731

            if X_block is not None:
                row = np.concatenate([y_feats_n, X_block[i - 1]])
            else:
                row = y_feats_n

            yhat_norm = float(
                np.asarray(self._estimator_.predict(row.reshape(1, -1))).ravel()[0]
            )
            yhat = float(np.asarray(inv(np.array([yhat_norm], dtype=float)))[0])

            preds[i - 1] = yhat
            last_roll = np.roll(last_roll, -1)
            last_roll[-1] = yhat

        # assemble Series for all 1..H steps, then subset to requested fh
        y_all = pd.Series(
            preds,
            index=abs_all_idx,
            name=self._y_name_ if self._y_name_ is not None else "y",
        )
        y_req = y_all.reindex(abs_req_idx)
        return y_req

    # -------------------- optional: update logic --------------------
    def _update(
        self, y: pd.Series, X: Optional[pd.DataFrame] = None, update_params: bool = True
    ):
        """Update forecaster with new data. If update_params=True, refit; else only roll window."""
        if (
            self._estimator_ is None
            or self._last_window_ is None
            or self._dir_estimators_ is None
        ):
            raise RuntimeError("Call fit(...) before update(...).")

        y = _ensure_series(y)
        if len(y) == 0:
            return self

        # roll last window with the new y values
        new_vals = y.to_numpy().astype(float).reshape(-1)
        if len(new_vals) >= self.window_length:
            self._last_window_ = new_vals[-self.window_length :]
        else:
            rolled = np.roll(self._last_window_, -len(new_vals))
            rolled[-len(new_vals) :] = new_vals
            self._last_window_ = rolled

        # refit if requested
        if update_params:
            # append to stored training data and refit from scratch (simple & robust)
            y_full = pd.concat([self._y_train_, y])
            X_full = None
            if self._X_train_ is not None:
                if X is None:
                    raise ValueError(
                        "This model was originally fit with X; update with matching X."
                    )
                X_full = pd.concat([self._X_train_, X]).sort_index()

            _check_regressor(self.estimator)

            # impute like in fit
            if self.impute_missing in ("ffill", "bfill"):
                y_imp = y_full.fillna(method=self.impute_missing)
            else:
                y_imp = y_full.copy()

            x_mode = self.x_mode
            if x_mode == "auto":
                x_mode = "concurrent" if X_full is not None else "none"

            # refit K heads
            dir_estimators: List[RegressorMixin] = []
            for h in range(1, self.steps_ahead + 1):
                Xt_h, yt_h = _build_supervised_table(
                    y=y_imp,
                    X=X_full,
                    window_length=self.window_length,
                    steps_ahead=h,
                    x_mode=x_mode,
                    normalization_strategy=self.normalization_strategy,
                )
                est_h = clone(self.estimator)
                est_h.fit(Xt_h.values, yt_h.values)
                dir_estimators.append(est_h)

            # update learned state
            self._dir_estimators_ = dir_estimators
            self._estimator_ = dir_estimators[0]
            self._x_used_ = (x_mode == "concurrent") and (X_full is not None)
            self._x_columns_ = list(X_full.columns) if (X_full is not None) else None
            self._y_train_index_ = y_full.index
            self._y_name_ = y_full.name
            self._y_train_ = y_full.copy()
            self._X_train_ = X_full.copy() if X_full is not None else None

            # refresh last window from y_full
            last = y_imp.iloc[-self.window_length :].to_numpy()
            self._last_window_ = last.astype(float).reshape(-1)

        return self

    # -------------------- fitted params exposure (optional) --------------------
    def _get_fitted_params(self):
        """Return fitted parameters."""
        return {
            "x_used": self._x_used_,
            "x_columns": self._x_columns_,
            "last_window": (
                None if self._last_window_ is None else self._last_window_.copy()
            ),
            "one_step_estimator": self._estimator_,
            "direct_estimators": self._dir_estimators_,
            "y_train_index": self._y_train_index_,
        }

    # -------------------- test params for sktime test suite --------------------
    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        """Return parameter settings for the estimator tests."""
        # import inside to keep soft deps contained in tests
        from sklearn.linear_model import LinearRegression, Ridge

        if parameter_set == "fast":
            return {
                "estimator": LinearRegression(),
                "window_length": 4,
                "steps_ahead": 2,
                "normalization_strategy": meanvar_window_normalizer,
            }

        return [
            {
                "estimator": LinearRegression(),
                "window_length": 5,
                "steps_ahead": 1,
                "normalization_strategy": None,
            },
            {
                "estimator": Ridge(alpha=0.1),
                "window_length": 3,
                "steps_ahead": 3,
                "normalization_strategy": meanvar_window_normalizer,
            },
        ]


# ---------------------------------------------------------------------------
# Convenience factory (matching the earlier API idea; optional)
# ---------------------------------------------------------------------------


def make_reduction(
    estimator: RegressorMixin,
    strategy: str = "recursive",
    window_length: int = 10,
    steps_ahead: Optional[int] = None,
    x_mode: str = "auto",
    impute_missing: Optional[str] = "bfill",
    normalization_strategy: NormStrategy = None,
) -> ReductionForecaster:
    """
    Construct a ReductionForecaster.

    In this unified design:
    - If ``strategy='recursive'`` and ``steps_ahead is None``, you'll get K=1 (pure recursive).
    - If ``strategy='direct'`` and you pass ``steps_ahead=K``, you'll get K direct heads
      for 1..K and recursive continuation beyond K.
    - Any other combination behaves the same as setting K=max(1, steps_ahead).
    - ``normalization_strategy`` may be provided either way and is applied per-window.

    Parameters
    ----------
    estimator : sklearn-style regressor
    strategy : {"recursive", "direct"}, default="recursive"
    window_length : int, default=10
    steps_ahead : int or None, default=None
        Number of direct heads (K). If None, K=1 for "recursive" and K=1 for "direct"
        unless explicitly provided.
    x_mode : {"auto","none","concurrent"}, default="auto"
    impute_missing : {"ffill","bfill",None}, default="bfill"
    normalization_strategy : callable or None, default=None

    Returns
    -------
    ReductionForecaster
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
        x_mode=x_mode,
        impute_missing=impute_missing,
        normalization_strategy=normalization_strategy,
    )


# ---------------------------------------------------------------------------
# Example normalization strategy
# ---------------------------------------------------------------------------


def meanvar_window_normalizer(
    window: np.ndarray,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """
    Return (transform, inverse_transform) based on the window's mean and std.

    Parameters
    ----------
    window : 1D ndarray
        The y-lag window in feature order (most recent first). Only its statistics
        are used; it is NOT modified in-place.

    Returns
    -------
    transform : f(arr_1d) -> arr_1d
        Applies (arr - mean) / max(std, eps)
    inverse_transform : f(arr_1d) -> arr_1d
        Applies arr * max(std, eps) + mean
    """
    w = np.asarray(window, dtype=float).ravel()
    mu = float(np.mean(w)) if w.size else 0.0
    sigma = float(np.std(w)) if w.size else 1.0
    # avoid division by zero
    scale = sigma if sigma > 0.0 else 1.0

    def transform(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=float).ravel()
        return (a - mu) / scale

    def inverse_transform(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=float).ravel()
        return a * scale + mu

    return transform, inverse_transform


# ---------------------------------------------------------------------------
# Minimal self-check (optional)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # tiny smoke test if run directly
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    n = 80
    t = pd.date_range("2023-01-01", periods=n, freq="D")
    y = pd.Series(
        np.sin(np.linspace(0, 6, n)) + 0.1 * rng.standard_normal(n), index=t, name="y"
    )
    X = pd.DataFrame({"cos": np.cos(np.linspace(0, 6, n))}, index=t)

    f = make_reduction(
        LinearRegression(),
        strategy="direct",
        window_length=7,
        steps_ahead=3,
        normalization_strategy=meanvar_window_normalizer,
    )
    f.fit(y, X=X)

    # make a future X for next H days
    H = 10
    fh = ForecastingHorizon(np.arange(1, H + 1), is_relative=True)
    # construct X rows for absolute fh timestamps
    abs_idx = _fh_to_absolute_index(
        fh.to_absolute(f.cutoff), cutoff=f.cutoff, y_index=y.index, H=H
    )
    Xf = pd.DataFrame({"cos": np.cos(np.linspace(6, 6 + 0.1 * H, H))}, index=abs_idx)

    print(f.predict(fh, X=Xf))
