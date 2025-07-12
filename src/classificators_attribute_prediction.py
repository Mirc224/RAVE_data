from attributes_prediction.classifier_attribute_predictors import ClassifierAttributePredictor
from models.rave_dataset import RaveSample
from utils.utility import load_rave_dataset
from models.response_model_creation import *
from constants.rave_constants import *
from experiments.running_experiments import AttributePredictionResult
import os
import json
from pathlib import Path
from cross_validation.fold_parameters_provider import FoldParametersProvider
from utils.attribute_predictor_utils import train

def load_experiment_data(file_path: str) -> list[dict]:
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r', encoding="utf8") as file:
        data = json.load(file)
    return data

def store_experiment_data(file_path: str, results: dict[dict]):
    with open(file_path, "w+", encoding="utf8") as f:
        json.dump(results, f)

used_version = "v2"
attribute_names_version = {
    "v1": ALL_SELECTED_ATTRIBUTES,
    "v2": ALL_SELECTED_ATTRIBUTES_V2,
}

RESULTS_FOLDER = 'results_attributes'
SELECTED_ATTRIBUTES = attribute_names_version[used_version]

print("Running attribute predictions using classificators!")
print(f"Dataset version: {used_version}")

already_runned_experiments: set = set()
for file_name in os.listdir(f"./{RESULTS_FOLDER}"):
    _, model_description = file_name.split("_", 1)
    model_description = model_description.removesuffix(".json")
    already_runned_experiments.add(model_description)

dataset_path = f"./dataset/rave_dataset_{used_version}.json"
rave_dataset = load_rave_dataset(dataset_path)
for sample in rave_dataset:
    sample.keep_selected_text_attributes(SELECTED_ATTRIBUTES)

SORTED_ATTRIBUTES = sorted(SELECTED_ATTRIBUTES)

folds_folder_path = Path("./dataset/folds/")
fold_parameters_provider = FoldParametersProvider()
fold_parameters_provider.folds_dict = fold_parameters_provider.read_folds_folder(folds_folder_path)
fold_parameters_provider.initialize_fold_data(rave_dataset)

fold_predictors: dict[str, list[ClassifierAttributePredictor]] = {}
for fold_number in fold_parameters_provider.fold_numbers:
    X_train, X_test = fold_parameters_provider.get_fold_train_test_data(fold_number)
    fold_predictors[fold_number] = list(train(X_train, SORTED_ATTRIBUTES))

run_parameters_combinations: list[tuple[list[RaveSample], ClassifierAttributePredictor, str]] = []
for fold_number in fold_parameters_provider.fold_numbers:
    _, fold_test_data = fold_parameters_provider.get_fold_train_test_data(fold_number)
    for predictor in fold_predictors[fold_number]:
        run_parameters_combinations.append((fold_test_data, predictor, fold_number))

for test_data, predictor, fold_number in run_parameters_combinations:
    experiment_name = f"{predictor.description}_{used_version}_{fold_number}"
    result_file = f"./{RESULTS_FOLDER}/{experiment_name}.json"
    
    result_dict_list: list[dict] = load_experiment_data(result_file)

    already_processed_ids = {result[AttributePredictionResult.ID] for result in result_dict_list}
    data_to_process = [sample for sample in test_data if sample.id not in already_processed_ids]

    if len(data_to_process) == 0:
        print(f"{experiment_name} already done.")
        continue

    print(f"Running {experiment_name}...")
    print(f"Number of samples to process: {len(data_to_process)}")

    checkpoint_to_store = int(len(data_to_process) * 0.10)

    result_dictionary = result_dict_list
        
    for data in data_to_process:
        result = AttributePredictionResult(data.id)
        result.predicted_attributes = list(predictor.predict([data]))[0]
        result_dict_list.append(result.to_dict())
        
        if len(result_dict_list) % STORE_AFTER == 0:
            store_experiment_data(result_file, result_dictionary)
        if len(result_dict_list) % checkpoint_to_store == 0:
            done_perc = round((len(result_dict_list) / len(test_data)) * 100)
            print(f"{experiment_name}: {done_perc}%")
        store_experiment_data(result_file, result_dictionary)