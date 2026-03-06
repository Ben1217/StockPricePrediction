"""
Model Trainer Module
Orchestrates model training, evaluation, and comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Type
from pathlib import Path
import json

from sklearn.model_selection import TimeSeriesSplit

from .base_model import BaseModel
from .xgboost_model import XGBoostModel
from .lstm_model import LSTMModel
from .random_forest_model import RandomForestModel
from ..utils.logger import get_logger

logger = get_logger(__name__)


MODEL_REGISTRY = {
    'xgboost': XGBoostModel,
    'lstm': LSTMModel,
    'random_forest': RandomForestModel,
}


class ModelTrainer:
    """Orchestrates model training and evaluation"""

    def __init__(self, models_dir: str = "models/saved_models"):
        """
        Initialize trainer

        Parameters
        ----------
        models_dir : str
            Directory for saving models
        """
        self.models_dir = Path(models_dir)
        self.models: Dict[str, BaseModel] = {}
        self.results: Dict[str, Dict] = {}

    def create_model(self, model_type: str, params: Optional[Dict] = None) -> BaseModel:
        """
        Create a model instance

        Parameters
        ----------
        model_type : str
            Type of model ('xgboost', 'lstm', 'random_forest')
        params : dict, optional
            Model parameters

        Returns
        -------
        BaseModel
            Model instance
        """
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(MODEL_REGISTRY.keys())}")

        return MODEL_REGISTRY[model_type](params)

    def train_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[Dict] = None,
        save: bool = True
    ) -> BaseModel:
        """
        Train a single model

        Parameters
        ----------
        model_type : str
            Type of model
        X_train, y_train : np.ndarray
            Training data
        X_val, y_val : np.ndarray, optional
            Validation data
        params : dict, optional
            Model parameters
        save : bool
            Whether to save the model

        Returns
        -------
        BaseModel
            Trained model
        """
        logger.info(f"Training {model_type} model...")

        model = self.create_model(model_type, params)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        self.models[model_type] = model

        if save:
            save_path = self.models_dir / model_type / f"{model_type}_model"
            model.save(str(save_path))

        return model

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        model_types: Optional[List[str]] = None,
        save: bool = True
    ) -> Dict[str, BaseModel]:
        """
        Train multiple models

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training data
        X_val, y_val : np.ndarray, optional
            Validation data
        model_types : list, optional
            Models to train (default: all)
        save : bool
            Whether to save models

        Returns
        -------
        dict
            Dictionary of trained models
        """
        if model_types is None:
            model_types = ['xgboost', 'random_forest']  # Exclude LSTM by default (slower)

        for model_type in model_types:
            try:
                self.train_model(model_type, X_train, y_train, X_val, y_val, save=save)
            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")

        return self.models

    def evaluate_all_models(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Evaluate all trained models

        Parameters
        ----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test targets

        Returns
        -------
        pandas.DataFrame
            Comparison of model metrics
        """
        results = []

        for name, model in self.models.items():
            try:
                metrics = model.evaluate(X_test, y_test)
                metrics['model'] = name
                results.append(metrics)
                self.results[name] = metrics
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index('model')
            df = df.sort_values('rmse')

        logger.info(f"Evaluation complete for {len(results)} models")
        return df

    def get_best_model(self, metric: str = 'rmse') -> Optional[BaseModel]:
        """
        Get the best performing model

        Parameters
        ----------
        metric : str
            Metric to compare (lower is better for rmse, mse, mae)

        Returns
        -------
        BaseModel or None
            Best model
        """
        if not self.results:
            logger.warning("No evaluation results available")
            return None

        # Find best model
        best_name = min(self.results, key=lambda x: self.results[x].get(metric, float('inf')))
        return self.models.get(best_name)

    def save_results(self, filepath: str) -> None:
        """Save evaluation results to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {filepath}")

    # ------------------------------------------------------------------
    # Walk-Forward (Rolling Window) Cross-Validation
    # ------------------------------------------------------------------

    def walk_forward_validate(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        gap: int = 5,
        params: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Walk-forward cross-validation with embargo gap.

        For each fold the model is trained from scratch on the training
        portion and evaluated on the test portion.  An embargo of *gap*
        samples is inserted between training and test sets to prevent
        information leakage from overlapping feature windows.

        Parameters
        ----------
        model_type : str
            One of 'xgboost', 'random_forest', 'lstm'.
        X : np.ndarray
            Full feature matrix (samples, features) or (samples, seq, feat).
        y : np.ndarray
            Full target vector.
        n_splits : int
            Number of walk-forward folds.
        gap : int
            Embargo period (number of samples) between train and test.
        params : dict, optional
            Model hyper-parameters (uses defaults if None).

        Returns
        -------
        pd.DataFrame
            Per-fold metrics with columns:
            fold, rmse, mae, r2, directional_accuracy, sharpe_ratio
            Plus a final 'mean ± std' summary row.
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        fold_metrics: List[Dict] = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(
                f"Walk-forward fold {fold_idx + 1}/{n_splits}: "
                f"train={len(train_idx)}, test={len(test_idx)}"
            )

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Use a fraction of training data as validation for early-stopping
            val_size = max(1, int(len(X_train) * 0.1))
            X_val, y_val = X_train[-val_size:], y_train[-val_size:]
            X_train_fold, y_train_fold = X_train[:-val_size], y_train[:-val_size]

            # Fresh model per fold
            model = self.create_model(model_type, params)
            try:
                model.fit(X_train_fold, y_train_fold, X_val=X_val, y_val=y_val)
            except TypeError:
                # Some models don't accept X_val / y_val
                model.fit(X_train_fold, y_train_fold)

            preds = model.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))

            # Directional accuracy
            if len(y_test) > 1:
                actual_dir = np.sign(y_test[1:] - y_test[:-1])
                pred_dir = np.sign(preds[1:] - preds[:-1])
                dir_acc = float(np.mean(actual_dir == pred_dir))
            else:
                dir_acc = 0.0

            sharpe = self._compute_sharpe(y_test, preds)

            fold_metrics.append({
                'fold': fold_idx + 1,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'directional_accuracy': dir_acc,
                'sharpe_ratio': sharpe,
            })

        df = pd.DataFrame(fold_metrics)

        # Append mean ± std summary row
        summary = {'fold': 'mean±std'}
        for col in ['rmse', 'mae', 'r2', 'directional_accuracy', 'sharpe_ratio']:
            summary[col] = f"{df[col].mean():.4f}±{df[col].std():.4f}"
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

        logger.info(f"Walk-forward validation complete ({n_splits} folds)")
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sharpe(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        periods_per_year: int = 252,
    ) -> float:
        """
        Compute an approximate annualised Sharpe Ratio from predicted
        returns vs actual returns.

        The 'strategy return' at each step is defined as:
            sign(predicted_return) * actual_return
        i.e. we take a long position when the model predicts positive
        and a short position when it predicts negative.

        Parameters
        ----------
        y_true : np.ndarray   Actual returns.
        y_pred : np.ndarray   Predicted returns.
        periods_per_year : int  Annualisation factor.

        Returns
        -------
        float   Annualised Sharpe Ratio (0.0 if degenerate).
        """
        strategy_returns = np.sign(y_pred) * y_true
        if len(strategy_returns) < 2 or np.std(strategy_returns) == 0:
            return 0.0
        return float(
            np.mean(strategy_returns)
            / np.std(strategy_returns)
            * np.sqrt(periods_per_year)
        )

    # ------------------------------------------------------------------
    # Optuna Hyperparameter Optimization
    # ------------------------------------------------------------------

    @staticmethod
    def _get_search_space(trial, model_type: str) -> Dict:
        """
        Define Optuna search space for each model type.

        Parameters
        ----------
        trial : optuna.Trial
        model_type : str

        Returns
        -------
        dict  Sampled hyperparameters for this trial.
        """
        if model_type == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'objective': 'reg:squarederror',
                'random_state': 42,
            }
        elif model_type == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42,
                'n_jobs': -1,
            }
        elif model_type == 'lstm':
            return {
                'units': trial.suggest_int('units', 32, 128, step=16),
                'layers': trial.suggest_int('layers', 1, 3),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'epochs': 50,       # capped for HPO speed
                'patience': 10,     # aggressive early stopping during HPO
                'sequence_length': 60,
            }
        else:
            raise ValueError(f"No search space defined for model: {model_type}")

    def optimize_hyperparameters(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
        n_cv_splits: int = 3,
        gap: int = 5,
    ) -> Tuple[Dict, "optuna.Study"]:
        """
        Run Optuna hyperparameter optimization maximising Sharpe Ratio.

        Each trial samples a set of hyperparameters, runs walk-forward CV
        over *n_cv_splits* folds, and reports the mean Sharpe across folds.

        Parameters
        ----------
        model_type : str
            One of 'xgboost', 'random_forest', 'lstm'.
        X : np.ndarray
            Full feature matrix.
        y : np.ndarray
            Full target vector (returns).
        n_trials : int
            Number of Optuna trials.
        n_cv_splits : int
            Number of walk-forward folds per trial.
        gap : int
            Embargo gap between train/test in each fold.

        Returns
        -------
        tuple
            (best_params: dict, study: optuna.Study)
        """
        import optuna
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_squared_error

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = self._get_search_space(trial, model_type)
            tscv = TimeSeriesSplit(n_splits=n_cv_splits, gap=gap)
            fold_sharpes: List[float] = []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Validation slice for early stopping
                val_size = max(1, int(len(X_train) * 0.1))
                X_val, y_val = X_train[-val_size:], y_train[-val_size:]
                X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]

                model = self.create_model(model_type, params)
                try:
                    model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
                except TypeError:
                    model.fit(X_tr, y_tr)

                preds = model.predict(X_test)
                fold_sharpes.append(self._compute_sharpe(y_test, preds))

            mean_sharpe = float(np.mean(fold_sharpes))

            # Report intermediate value for pruning
            trial.report(mean_sharpe, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return mean_sharpe

        study = optuna.create_study(
            direction='maximize',
            study_name=f'{model_type}_hpo',
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        logger.info(
            f"Optuna HPO complete for {model_type}: "
            f"best Sharpe={study.best_value:.4f}, "
            f"params={best_params}"
        )

        return best_params, study
