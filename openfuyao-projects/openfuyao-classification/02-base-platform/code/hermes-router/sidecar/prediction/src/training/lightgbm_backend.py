# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""LightGBM quantile backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default hyperparameters mirror the spec baseline (Training Backend And Hyperparameter Baseline).
# Override individual entries through TrainOptions.hyperparams or the trainer CLI --hyperparams flag.
DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "force_col_wise": True,
    "random_state": 42,
    "verbosity": -1,
}


def fit_quantile_model(
    features: np.ndarray,
    labels: np.ndarray,
    quantile: float,
    overrides: dict[str, Any] | None = None,
    monotone_constraints: tuple[int, ...] | None = None,
):
    import lightgbm as lgb

    params: dict[str, Any] = {
        "objective": "quantile",
        "alpha": quantile,
        **DEFAULT_HYPERPARAMS,
    }
    if monotone_constraints is not None:
        # LightGBM's quantile objective rejects monotone_constraints outright, so
        # ignore them here rather than fail the run; the XGBoost backend is the one
        # that enforces monotonicity. The parameter is kept for a symmetric API.
        logger.warning("LightGBM quantile objective does not support monotone_constraints; ignoring")
    if overrides:
        params.update(overrides)
    model = lgb.LGBMRegressor(**params)
    model.fit(features, labels)
    return model.booster_


def predict(model, features: np.ndarray) -> np.ndarray:
    return model.predict(features)


def save_model(model, path: Path) -> None:
    model.save_model(str(path))
