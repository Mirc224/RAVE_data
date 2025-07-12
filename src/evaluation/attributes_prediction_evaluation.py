from constants.rave_constants import ALL_SELECTED_ATTRIBUTES_V2
import itertools
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from constants.metrics_constants import *

def calculate_accuracy(metrics_dict: dict) -> float:
    tp = metrics_dict[TP_KEY]
    fp = metrics_dict[FP_KEY]
    tn = metrics_dict[TN_KEY]
    fn = metrics_dict[FN_KEY]
    return (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

def calculate_precision(metrics_dict: dict) -> float:
    tp = metrics_dict[TP_KEY]
    fp = metrics_dict[FP_KEY]
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def calculate_recall(metrics_dict: dict) -> float:
    tp = metrics_dict[TP_KEY]
    fn = metrics_dict[FN_KEY]
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def calculate_f1(metric_dict: dict):
    precision = calculate_precision(metric_dict)
    recall = calculate_recall(metric_dict)
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def get_default_metrics_dict() -> dict[str, float]:
    metrics_to_display = [prefix + metric for prefix, metric in itertools.product(SCOPE_PREFIXES, EVALUATED_METRICS)]
    metrics_to_display.append(JACCARD_KEY)
    return {metric: 0.0 for metric in metrics_to_display}

def evaluate_metrics_using_cm(metric_dict: dict, prefix: str = "") -> dict:
    return {
        f"{prefix}{ACCURACY_KEY}" : calculate_accuracy(metric_dict),
        f"{prefix}{PRECISION_KEY}" : calculate_precision(metric_dict),
        f"{prefix}{RECALL_KEY}": calculate_recall(metric_dict),
        f"{prefix}{F1_KEY}": calculate_f1(metric_dict)
    }

def evaluate_macro_metrics(total_metrics: dict, number_of_classes: int, prefix: str = "") -> dict:
    return {
        f"{prefix}{ACCURACY_KEY}": total_metrics[ACCURACY_KEY] / number_of_classes,
        f"{prefix}{PRECISION_KEY}": total_metrics[PRECISION_KEY] / number_of_classes,
        f"{prefix}{RECALL_KEY}": total_metrics[RECALL_KEY] / number_of_classes,
        f"{prefix}{F1_KEY}": total_metrics[F1_KEY] / number_of_classes,
    }

def evaluate_sample_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    precision_list: list[float] = []
    recall_list: list[float] = []
    f1_list: list[float] = []
    accuaracy_list: list[float] = []

    for i in range(y_test.shape[0]):
        sample_dict: dict[str, int] = {}
        sample_dict[TP_KEY] = np.sum((y_pred[i] == 1) & (y_test[i] == 1))  # True Positives
        sample_dict[FP_KEY] = np.sum((y_pred[i] == 1) & (y_test[i] == 0))  # False Positives
        sample_dict[FN_KEY] = np.sum((y_pred[i] == 0) & (y_test[i] == 1))  # False Negatives
        precision_list.append(calculate_precision(sample_dict))
        recall_list.append(calculate_recall(sample_dict))
        f1_list.append(calculate_f1(sample_dict))
        accuaracy_list.append(np.array_equal(y_pred[i], y_test[i]))
    return {
        f"{SAMPLE_PREXIF}{PRECISION_KEY}" : float(np.mean(precision_list)),
        f"{SAMPLE_PREXIF}{RECALL_KEY}" : float(np.mean(recall_list)),
        f"{SAMPLE_PREXIF}{F1_KEY}" : float(np.mean(f1_list)),
        f"{SAMPLE_PREXIF}{ACCURACY_KEY}": float(np.mean(accuaracy_list)),
        JACCARD_KEY: float(jaccard_score(y_test, y_pred, average="samples", zero_division=0))
    }
    
def get_confustion_matrix_dict(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    if cm.size == 1:
        cm_value = cm[0, 0]
        cm = np.array([[0, 0], [0, 0]])
        max_val = np.max(y_test)
        cm[max_val, max_val] = cm_value
    return {
        TN_KEY: int(cm[0, 0]),
        FP_KEY: int(cm[0, 1]),
        FN_KEY: int(cm[1, 0]),
        TP_KEY: int(cm[1, 1])
    }

def calculate_weighted_metrics(y_true: np.ndarray, evaluated_metrics: dict, number_of_classes: int) -> dict:
    attr_counts = {attribute : sum(y_true[:, i]) for i, attribute in enumerate(range(number_of_classes))}
    total_count = sum(attr_counts.values())
    return {
        f"{WEIGHTED_PREFIX}{ACCURACY_KEY}": float(sum([evaluated_metrics[attribute][ACCURACY_KEY] * attr_counts[attribute] for attribute in range(number_of_classes)]) / total_count),
        f"{WEIGHTED_PREFIX}{PRECISION_KEY}": float(sum([evaluated_metrics[attribute][PRECISION_KEY] * attr_counts[attribute] for attribute in range(number_of_classes)]) / total_count),
        f"{WEIGHTED_PREFIX}{RECALL_KEY}": float(sum([evaluated_metrics[attribute][RECALL_KEY] * attr_counts[attribute] for attribute in range(number_of_classes)]) / total_count),
        f"{WEIGHTED_PREFIX}{F1_KEY}": float(sum([evaluated_metrics[attribute][F1_KEY] * attr_counts[attribute] for attribute in range(number_of_classes)]) / total_count)
    }

def evaluate_predictions(y_test: pd.DataFrame, y_pred: np.ndarray) -> dict:
    evaluation = {}
    total_metrics = {
        ACCURACY_KEY : 0,
        TN_KEY: 0,
        FP_KEY: 0,
        FN_KEY: 0,
        TP_KEY: 0,
        ACCURACY_KEY : 0,
        PRECISION_KEY : 0,
        RECALL_KEY : 0,
        F1_KEY : 0,
    }
    number_of_classes = y_test.shape[1]
    for i, attribute in enumerate(range(number_of_classes)):
        y_gold = y_test[:, i]
        y_pred_attribute = y_pred[:, i]
        attribute_metrics_evaluated = get_confustion_matrix_dict(y_gold, y_pred_attribute) | {
            ACCURACY_KEY: accuracy_score(y_gold, y_pred_attribute),
            PRECISION_KEY: float(precision_score(y_gold, y_pred_attribute, zero_division=0)),
            RECALL_KEY: float(recall_score(y_gold, y_pred_attribute, zero_division=0)),
            F1_KEY: float(f1_score(y_gold, y_pred_attribute, zero_division=0)),
            "support": int(np.sum(y_gold))
        }
        evaluation[attribute] = attribute_metrics_evaluated
        for metric_name in TOTAL_METRICS_NAMES:
            total_metrics[metric_name] += attribute_metrics_evaluated[metric_name]
    return evaluate_metrics_using_cm(total_metrics, MICRO_PREFIX) | \
            evaluate_macro_metrics(total_metrics, number_of_classes, MACRO_PREFIX) | \
            calculate_weighted_metrics(y_test, evaluation, number_of_classes) | \
            evaluate_sample_metrics(y_test, y_pred) | \
            {"metrics_per_attribute": evaluation}

def evaluate_encoding_run(encoding_results: list[dict], model_names: list[str]) -> dict[str, dict[str, float]]:
    result_dict = { model_name: get_default_metrics_dict() for model_name in model_names }
    number_of_runs = 0
    for run_results in encoding_results:
        number_of_runs += 1
        for model_name, total_model_results in result_dict.items():
            model_run_results = run_results[model_name]
            for metric_name in total_model_results.keys():
                total_model_results[metric_name] += model_run_results[metric_name]

    for model_results in result_dict.values():
        for metric_name, metric_value in model_results.items():
            model_results[metric_name] = metric_value / number_of_runs
    return result_dict