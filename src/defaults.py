"""Shared application defaults."""

DEFAULT_INDEX_SYMBOL = "^GSPC"

# One switch governs both skill gates -- the regression baseline check in
# src.models.ensemble_predictor and the direction check in
# src.models.direction_utils. It lives here so neither module has to import the
# other just to agree on the name.
ENFORCE_MODEL_SKILL_ENV = "QUANTVISION_ENFORCE_MODEL_SKILL"

# Five years of daily bars, matching what the price-regression bundles already
# train on. Measured against the previous 756: across 11 symbols the tree models
# clear the direction skill gate 18 times out of 22 at this window versus 12 at
# the shorter one, and the per-symbol ROC-AUC is both higher and less erratic.
DEFAULT_TRAINING_LOOKBACK_DAYS = 1825
