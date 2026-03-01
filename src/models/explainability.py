"""
Model Explainability Module.
SHAP-based feature importance for tree models (XGBoost, RF).
Gradient-based attribution for LSTM.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


def compute_shap_importance(
    model,
    X_sample: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_k: int = 20,
) -> Dict:
    """
    Compute SHAP feature importance for tree-based models.

    Parameters
    ----------
    model : fitted model object (XGBoost or RandomForest .model attribute)
    X_sample : np.ndarray
        Sample of feature data (use ~100-500 rows)
    feature_names : list, optional
        Names of features
    top_k : int
        Number of top features to return

    Returns
    -------
    dict
        {feature_name: importance_value, ...} sorted descending
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed, using fallback importance")
        return _fallback_importance(model, feature_names, top_k)

    try:
        # Use TreeExplainer for tree models
        if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            inner = model.model
        else:
            inner = model

        explainer = shap.TreeExplainer(inner)
        shap_values = explainer.shap_values(X_sample[:min(200, len(X_sample))])

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        mean_abs = np.abs(shap_values).mean(axis=0)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(mean_abs))]

        importance = dict(zip(feature_names, mean_abs.tolist()))
        sorted_imp = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        )

        logger.info(f"Computed SHAP importance for {len(sorted_imp)} features")
        return {
            "method": "shap_tree",
            "features": sorted_imp,
            "total_features": len(importance),
        }
    except Exception as e:
        logger.warning(f"SHAP failed: {e}, using fallback")
        return _fallback_importance(model, feature_names, top_k)


def compute_gradient_importance(
    model,
    X_sample: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_k: int = 20,
) -> Dict:
    """
    Gradient-based feature attribution for PyTorch LSTM.

    Parameters
    ----------
    model : LSTMModel instance
    X_sample : np.ndarray
        Sample input (samples, seq_len, features)
    """
    import torch

    if len(X_sample.shape) == 2:
        X_sample = X_sample.reshape((X_sample.shape[0], X_sample.shape[1], 1))

    X_t = torch.FloatTensor(X_sample[:min(50, len(X_sample))]).requires_grad_(True)

    if hasattr(model, 'model'):
        net = model.model
    else:
        net = model

    net.eval()
    output = net(X_t)
    output.sum().backward()

    grads = X_t.grad.abs().mean(dim=(0, 1)).detach().numpy()  # mean over samples and timesteps

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(grads))]

    importance = dict(zip(feature_names[:len(grads)], grads.tolist()))
    sorted_imp = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
    )

    return {
        "method": "gradient_attribution",
        "features": sorted_imp,
        "total_features": len(importance),
    }


def _fallback_importance(model, feature_names, top_k) -> Dict:
    """Fallback: use scikit-learn feature_importances_ if available."""
    inner = model.model if hasattr(model, 'model') else model
    if hasattr(inner, 'feature_importances_'):
        imp = inner.feature_importances_
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(imp))]
        importance = dict(zip(feature_names, imp.tolist()))
        sorted_imp = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        )
        return {"method": "sklearn_importance", "features": sorted_imp, "total_features": len(importance)}
    return {"method": "unavailable", "features": {}, "total_features": 0}


def get_model_explainability(
    model,
    X_sample: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_k: int = 20,
) -> Dict:
    """
    Auto-detect model type and compute appropriate explainability.
    """
    model_name = getattr(model, 'name', type(model).__name__).lower()

    if 'lstm' in model_name:
        return compute_gradient_importance(model, X_sample, feature_names, top_k)
    else:
        return compute_shap_importance(model, X_sample, feature_names, top_k)
