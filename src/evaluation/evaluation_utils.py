import os
import json
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from models.rave_dataset import RaveSample
from evaluation.experiment_evaluation import ExperimentResults, ExperimentEvaluator

BASE_COLUMNS = [
    ExperimentResults.MODEL_NAME, ExperimentResults.MODEL_NAME_W_SIZE, ExperimentResults.MODEL_NORMALIZED_SIZE, ExperimentResults.MODEL_WITH_EXPERIMENT,ExperimentResults.MODEL_SIZE, 
    ExperimentResults.EXPERIMENT_NAME, ExperimentResults.EXPERIMENT_CODE_NAME, ExperimentResults.DATASET_VERSION, ExperimentResults.ORACLE_ATTRIBUTES, ExperimentResults.ORACLE_SENTENCES, 
    ExperimentResults.ORACLE_TYPES, ExperimentResults.ADDITIONAL_DESCRIPTIONS, ExperimentResults.ATTRIBUTE_PREDICTION, ExperimentResults.ATTRIBUTE_PREDICTION_MODEL,
    ExperimentResults.ENCODING_METHOD, ExperimentResults.MODEL_FAMILY_COLOR
]

METRICS_COLUMNS = [
    *[f"{comb[0]}{comb[1]}" for comb in itertools.product(["micro_", "macro_"], [ExperimentEvaluator.F1_SCORE, ExperimentEvaluator.PRECISION, ExperimentEvaluator.RECALL])],
    ExperimentEvaluator.VALID_OUTPUTS_COUNT, ExperimentEvaluator.INVALID_OUTPUTS_COUNT, ExperimentEvaluator.AVERAGE_RESPONSE_TIME, ExperimentEvaluator.AVERAGE_RETRIES
]

def read_result_file(result_file_path: Path) -> dict[str, dict|str|list]:
    with open(result_file_path, "r", encoding="utf8") as f:
        return json.load(f)
    
def read_all_results_in_folder(results_file_path: Path) -> dict[str, dict]:
    all_results: dict[str, dict] = {}
    for file in os.listdir(results_file_path):
        result_file = results_file_path / file
        experiment_name = os.path.splitext(file)[0]
        all_results[experiment_name] = read_result_file(result_file)
    return all_results

def create_experiment_results_from_all_results(all_results: dict[str, dict]) -> dict[str, ExperimentResults]:
    all_experiment_results: dict[str, ExperimentResults] = {}
    for experiment_name, experiment_results in all_results.items():
        experiment_result = ExperimentResults(experiment_name, experiment_results)
        all_experiment_results[experiment_result.experiment_name] = experiment_result
    return all_experiment_results

def evaluate_experiments(rave_dataset: list[RaveSample], all_experiment_results: dict[str, ExperimentResults], selected_attributes: list[str]) -> dict[str, dict]:
    experiment_evaluator = ExperimentEvaluator(rave_dataset, selected_attributes)
    return { experiment_name: experiment_evaluator.evaluate_experiment(experiment_result) for experiment_name, experiment_result in all_experiment_results.items()}

def evaluate_k_fold(data: pd.DataFrame) -> pd.DataFrame:
    base_df = data[BASE_COLUMNS].drop_duplicates(subset=[ExperimentResults.MODEL_WITH_EXPERIMENT])

    prefixes = ["micro_", "macro_"]
    metrics_column = [
        *[f"{comb[0]}{comb[1]}" for comb in itertools.product(prefixes, [ExperimentEvaluator.F1_SCORE, ExperimentEvaluator.PRECISION, ExperimentEvaluator.RECALL])],
        ExperimentEvaluator.VALID_OUTPUTS_COUNT, ExperimentEvaluator.INVALID_OUTPUTS_COUNT, ExperimentEvaluator.AVERAGE_RESPONSE_TIME, ExperimentEvaluator.AVERAGE_RETRIES
    ]
    mean_df = data.groupby(ExperimentResults.MODEL_WITH_EXPERIMENT)[metrics_column].mean().reset_index()
    return base_df.merge(mean_df, on=ExperimentResults.MODEL_WITH_EXPERIMENT, how="inner")

def get_experiment_results_df(rave_dataset: list[RaveSample], all_results: list[str, dict], selected_attributes: list[str]) -> pd.DataFrame:
    all_experiment_results: dict[str, ExperimentResults] = create_experiment_results_from_all_results(all_results)
    evaluated_experiments: dict[str, dict] = evaluate_experiments(rave_dataset, all_experiment_results, selected_attributes)

    metrics_to_extract = [ExperimentEvaluator.PRECISION, ExperimentEvaluator.RECALL, ExperimentEvaluator.F1_SCORE]
    perspective_name = ["micro", "macro"]
    scoped_metrics = [f"{scope}_{metric}" for scope, metric in itertools.product(perspective_name, metrics_to_extract)]
    experiment_result_dict_list: list[dict] = []

    for experiment in all_experiment_results.values():
        experiment_result = experiment.experiment_configuration
        experiment_evaluation = evaluated_experiments[experiment.experiment_name]
        experiment_result[ExperimentEvaluator.INVALID_OUTPUTS_COUNT] = experiment_evaluation[ExperimentEvaluator.INVALID_OUTPUTS_COUNT]
        experiment_result[ExperimentEvaluator.VALID_OUTPUTS_COUNT] = experiment_evaluation[ExperimentEvaluator.VALID_OUTPUTS_COUNT]
        experiment_result[ExperimentEvaluator.AVERAGE_RESPONSE_TIME] = experiment_evaluation[ExperimentEvaluator.AVERAGE_RESPONSE_TIME]
        experiment_result[ExperimentEvaluator.AVERAGE_RETRIES] = experiment_evaluation[ExperimentEvaluator.AVERAGE_RETRIES]
        for metric in scoped_metrics:
            experiment_result[metric] = experiment_evaluation[metric]
        experiment_result["resultObject"] = experiment_evaluation
        experiment_result_dict_list.append(experiment_result)

    experiment_results_df: pd.DataFrame = pd.DataFrame(data=experiment_result_dict_list)
    family_color_mapping = {
        "gemma2": plt.cm.tab20.colors[0],
        "gemma": plt.cm.tab20.colors[1],
        "llama3.2": plt.cm.tab20.colors[2],
        "llama3.1": plt.cm.tab20.colors[3],
        "qwen2.5" : plt.cm.tab20.colors[4],
        "qwen2" : plt.cm.tab20.colors[5],
        "mistral": plt.cm.tab20.colors[6],
        "phi3.5" : plt.cm.tab20c.colors[16],
        "phi3" : plt.cm.tab20c.colors[17],
        "phi" : plt.cm.tab20c.colors[18],

    }
    experiment_results_df[ExperimentResults.MODEL_FAMILY_COLOR] = experiment_results_df[ExperimentResults.MODEL_NAME].map(family_color_mapping)
    return experiment_results_df