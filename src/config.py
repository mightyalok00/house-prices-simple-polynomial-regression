"""Single source of truth for reproducible modeling settings."""

RANDOM_STATE = 42
CANDIDATE_DEGREES = (1, 2)
CV_SPLITS = 5
CV_REPEATS = 5
HOLDOUT_SIZE = 0.20
BOOTSTRAP_RESAMPLES = 2_000
SELECTION_METRIC = "CV_RMSE"
