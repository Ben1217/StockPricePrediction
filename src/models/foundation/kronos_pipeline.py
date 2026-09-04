import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import sys
import logging

from ..kronos_direction import KronosDirection

logger = logging.getLogger(__name__)

class KronosPipeline:
    """
    Dedicated pipeline for Kronos, the sequence foundation model.
    Consumes raw OHLCV only.
    Output: N sampled future candle paths (default 128).
    Price = median of final closes.
    P(up) = fraction of sampled paths exceeding the last observed close.
    """
    def __init__(self, sample_count=128, lookback=128):
        self.name = "kronos"
        self.sample_count = sample_count
        self.lookback = lookback
        self.model = None
        
    def _ensure_model(self):
        if self.model is None:
            self.model = KronosDirection(
                sample_count=self.sample_count,
                lookback=self.lookback
            )
        return self.model

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> Dict[str, Any]:
        """
        Predict future price, probability of up move, and return samples.
        
        Args:
            df: OHLCV dataframe up to the current day.
            horizon: forecast horizon. KronosDirection currently supports 1-step natively,
                     for multi-step it autoregresses internally. Here we use it for 1-step.
                     
        Returns:
            Dict containing:
            - 'price': Predicted future price (median of samples)
            - 'p_up': Probability of price going up
            - 'samples': The raw samples (N,)
            - 'quantiles': Quantiles derived from samples
        """
        if df.empty or len(df) < self.lookback:
            raise ValueError(f"Dataframe must have at least {self.lookback} rows of OHLCV")
            
        model = self._ensure_model()
        model.set_ohlcv_context(df)
        
        # We need a dummy label array for the single row to satisfy the DirectionEstimator API
        # but we bypass fit and just call the raw prediction method directly
        last_date = df.index[-1]
        
        # _prepare_row and _sample_chunk are internal but they bypass the fit loop
        prepared = model._prepare_row(last_date)
        if not prepared:
            raise ValueError("Failed to prepare row for Kronos")
            
        predictor = model._ensure_predictor()
        sampled = model._sample_chunk(predictor, [prepared])[0]
        
        last_close = prepared["last_close"]
        p_up = float(np.mean(sampled > last_close))
        median_price = float(np.median(sampled))
        
        quantiles = {
            q: float(np.percentile(sampled, int(q*100))) 
            for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        }
        
        return {
            "price": median_price,
            "p_up": p_up,
            "samples": sampled,
            "quantiles": quantiles
        }
