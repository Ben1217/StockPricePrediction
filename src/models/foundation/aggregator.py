import numpy as np
from typing import Dict, Any, List

class FoundationAggregator:
    """
    Aggregates predictions from the foundation models using inverse-variance weighting.
    """
    
    @staticmethod
    def aggregate(predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple model predictions.
        
        Args:
            predictions: Dict mapping model_name -> prediction_result
                prediction_result contains 'price', 'p_up', 'quantiles' (or 'samples')
                
        Returns:
            Dict with aggregated 'price', 'p_up'
        """
        if not predictions:
            raise ValueError("No predictions to aggregate")
            
        weights: Dict[str, float] = {}
        skipped: Dict[str, str] = {}
        total_weight = 0.0
        
        # Calculate variance for each model
        # We estimate variance from the 10th and 90th quantiles (or samples)
        # Var ~ ((P90 - P10) / 2.56)^2 assuming rough normality for the spread
        for name, result in predictions.items():
            # A quantile model reports `samples: None` -- the key is PRESENT and
            # the value is not a sequence. Testing `'samples' in result` therefore
            # took this branch and raised TypeError on len(None), which took the
            # whole ensemble down as soon as a quantile model joined it. What
            # matters is whether there are samples, not whether the key exists.
            samples = result.get('samples')
            quantiles = result.get('quantiles')
            if samples is not None and len(samples) > 1:
                variance = np.var(samples)
            elif quantiles:
                # Try to get 0.1 and 0.9, or default to 0.2/0.8 etc.
                p90 = quantiles.get(0.9, quantiles.get(0.95, quantiles.get(0.8)))
                p10 = quantiles.get(0.1, quantiles.get(0.05, quantiles.get(0.2)))
                if p90 is not None and p10 is not None:
                    # rough approximation of variance
                    variance = ((p90 - p10) / 2.56) ** 2
                else:
                    variance = 1e-6
            else:
                variance = None

            # A model with no usable spread cannot be inverse-variance weighted.
            # The previous code substituted variance = 1e-6 and floored it at
            # 1e-8, giving that model a weight of up to 1e8 -- so the ONE model
            # that failed to report a distribution silently dominated the
            # ensemble completely. Such models are excluded and reported instead.
            if variance is None or not np.isfinite(variance) or variance <= 0:
                skipped[name] = "no usable forecast spread to weight by"
                continue

            weight = 1.0 / variance
            weights[name] = weight
            total_weight += weight
            
        if not weights or total_weight <= 0:
            # Falling back to equal weights is explicitly a DIFFERENT method
            # from the one Requirement 5.1 names, so it is labelled as such
            # rather than presented as inverse-variance weighting.
            method = "equal_weight_fallback"
            weights = {name: 1.0 / len(predictions) for name in predictions}
        else:
            method = "inverse_variance"
            for name in weights:
                weights[name] /= total_weight

        # Requirement 5.1: aggregate the PROBABILITIES and threshold at 0.5.
        # Never vote on the UP/DOWN labels, and never sign(price - close).
        agg_price = sum(weights[n] * predictions[n]['price'] for n in weights)
        agg_p_up = sum(weights[n] * predictions[n]['p_up'] for n in weights)

        return {
            "price": agg_price,
            "p_up": agg_p_up,
            "weights": weights,
            "direction": 1 if agg_p_up > 0.5 else 0,
            # Requirement 5.1: expose which aggregation method is active.
            "method": method,
            "models_used": sorted(weights),
            "models_excluded": skipped,
        }
