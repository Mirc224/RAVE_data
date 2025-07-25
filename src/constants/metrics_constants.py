TN_KEY = "tn"
FP_KEY = "fp"
FN_KEY = "fn"
TP_KEY = "tp"
ACCURACY_KEY = "accuracy"
PRECISION_KEY = "precision"
RECALL_KEY = "recall"
JACCARD_KEY = "jaccard"
F1_KEY = "f1"
TOTAL_METRICS_NAMES = [TN_KEY, FP_KEY, FN_KEY, TP_KEY, ACCURACY_KEY, PRECISION_KEY, RECALL_KEY, F1_KEY]
WEIGHTED_PREFIX = "weighted_"
MICRO_PREFIX = "micro_"
MACRO_PREFIX = "macro_"
SAMPLE_PREXIF = "sample_"

SCOPE_PREFIXES = [MICRO_PREFIX, MACRO_PREFIX, WEIGHTED_PREFIX, SAMPLE_PREXIF]
EVALUATED_METRICS = [PRECISION_KEY, RECALL_KEY, F1_KEY, ACCURACY_KEY]

MICRO_ACCURACY_KEY = MICRO_PREFIX + ACCURACY_KEY
MICRO_PRECISION_KEY = MICRO_PREFIX + PRECISION_KEY
MICRO_RECALL_KEY = MICRO_PREFIX + RECALL_KEY
MICRO_JACCARD_KEY = MICRO_PREFIX + JACCARD_KEY
MICRO_F1_KEY = MICRO_PREFIX + F1_KEY

MACRO_ACCURACY_KEY = MACRO_PREFIX + ACCURACY_KEY
MACRO_PRECISION_KEY = MACRO_PREFIX + PRECISION_KEY
MACRO_RECALL_KEY = MACRO_PREFIX + RECALL_KEY
MACRO_JACCARD_KEY = MACRO_PREFIX + JACCARD_KEY
MACRO_F1_KEY = MACRO_PREFIX + F1_KEY

WEIGHTED_ACCURACY_KEY = WEIGHTED_PREFIX + ACCURACY_KEY
WEIGHTED_PRECISION_KEY = WEIGHTED_PREFIX + PRECISION_KEY
WEIGHTED_RECALL_KEY = WEIGHTED_PREFIX + RECALL_KEY
WEIGHTED_JACCARD_KEY = WEIGHTED_PREFIX + JACCARD_KEY
WEIGHTED_F1_KEY = WEIGHTED_PREFIX + F1_KEY

SAMPLE_ACCURACY_KEY = SAMPLE_PREXIF + ACCURACY_KEY
SAMPLE_PRECISION_KEY = SAMPLE_PREXIF + PRECISION_KEY
SAMPLE_RECALL_KEY = SAMPLE_PREXIF + RECALL_KEY
SAMPLE_JACCARD_KEY = SAMPLE_PREXIF + JACCARD_KEY
SAMPLE_F1_KEY = SAMPLE_PREXIF + F1_KEY