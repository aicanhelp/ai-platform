# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""XGBoost quantile backend."""

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
    "min_child_weight": 5,
    "gamma": 0.2,
    "reg_alpha": 0.01,
    "reg_lambda": 0.1,
    "tree_method": "hist",
    "enable_categorical": True,
    "random_state": 42,
}


def fit_quantile_model(
    features: np.ndarray,
    labels: np.ndarray,
    quantile: float,
    overrides: dict[str, Any] | None = None,
    monotone_constraints: tuple[int, ...] | None = None,
):
    import xgboost as xgb

    params: dict[str, Any] = {
        "objective": "reg:quantileerror",
        "quantile_alpha": quantile,
        **DEFAULT_HYPERPARAMS,
    }
    if monotone_constraints is not None:
        params["monotone_constraints"] = tuple(monotone_constraints)
    if overrides:
        # Explicit overrides win, including monotone_constraints; warn rather than
        # silently merging two competing constraint specs.
        if monotone_constraints is not None and "monotone_constraints" in overrides:
            logger.warning("monotone_constraints override replaces the schema-derived constraints")
        params.update(overrides)
    model = xgb.XGBRegressor(**params)
    model.fit(features, labels)
    return model.get_booster()


def predict(model, features: np.ndarray) -> np.ndarray:
    import xgboost as xgb

    return model.predict(xgb.DMatrix(features))


def save_model(model, path: Path) -> None:
    model.save_model(str(path))
