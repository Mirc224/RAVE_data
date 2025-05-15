from utility import get_attribute_types_according_to_data, load_rave_dataset, pull_ollama_models, load_experiment_data, store_experiment_data
from response_model_creation import *
from rave_constants import *
from running_experiments import LLMOracleExperimentRunner, SampleProcessingResult
from datetime import datetime
import os
import json
from pydantic import BaseModel
from pathlib import Path
from fold_parameters_provider import FoldParametersProvider
from rave_dataset import RaveSample
from text_attributes_predictor import TextAttributesPredictor
from multilabel_classificators import XGBoostMultiClassifier, LogisticRegressionMultiClassifier, SVMMultiClassifier, KNNClassifierBase
from vectorizers import BoWVectorizer, TFIDFVectorizer, EmbeddingsVectorizer

used_version = "v2"
attribute_names_version = {
    "v1": ALL_SELECTED_ATTRIBUTES,
    "v2": ALL_SELECTED_ATTRIBUTES_V2,
}

SELECTED_ATTRIBUTES = attribute_names_version[used_version]

print("Running RAVE dataset experiments!")
print(f"Dataset version: {used_version}")

MODELS_TO_TEST = [
    "llama3.2:1b",
    "llama3.2:3b",
    "llama3.1:8b",
    "mistral:7b",
    "gemma:2b",
    "gemma:7b",
    "gemma2:2b",
    "gemma2:9b",
    "gemma2:27b",
    "qwen2:0.5b",
    "qwen2:1.5b",
    "qwen2:7b",
    "qwen2.5:0.5b",
    "qwen2.5:1.5b",
    "qwen2.5:3b",
    "qwen2.5:7b",
    "qwen2.5:14b",
    # "phi:2.7b",
    # "phi3:3.8b",
    # "phi3.5:3.8b",
    # "phi3:14b",
    "qwen2.5:32b",
]

pull_ollama_models(MODELS_TO_TEST)

already_runned_experiments: set = set()
for file_name in os.listdir("./results"):
    _, model_description = file_name.split("_", 1)
    model_description = model_description.removesuffix(".json")
    already_runned_experiments.add(model_description)

dataset_path = f"./dataset/rave_dataset_{used_version}.json"
rave_dataset = load_rave_dataset(dataset_path)
for sample in rave_dataset:
    sample.keep_selected_text_attributes(SELECTED_ATTRIBUTES)

attribute_enhancements_path = f"./schema_enhancement/attributes_descriptions_{used_version}.json"
with open(attribute_enhancements_path, "r", encoding="utf8") as f:
    loaded_attribute_descriptions = json.load(f)

attributes_types_dict: dict[str, list[type]] = get_attribute_types_according_to_data(rave_dataset)

SORTED_ATTRIBUTES = sorted(SELECTED_ATTRIBUTES)

folds_folder_path = Path("./dataset/folds/")
fold_parameters_provider = FoldParametersProvider()
fold_parameters_provider.folds_dict = fold_parameters_provider.read_folds_folder(folds_folder_path)
fold_parameters_provider.initialize_fold_data(rave_dataset)
fold_parameters_provider.train_for_each_fold(SORTED_ATTRIBUTES)

run_parameters_combinations = []
fold_numbers = fold_parameters_provider.fold_numbers

for fold_number in fold_parameters_provider.fold_numbers:
    _, fold_test_data = fold_parameters_provider.get_fold_train_test_data(fold_number)
    run_parameters_combinations.append((fold_test_data,{ 'oracle_sentences': False, 'oracle_attributes': True, 'oracle_types': False }, ({}, {}), None, fold_number))
    run_parameters_combinations.append((fold_test_data,{ 'oracle_sentences': False, 'oracle_attributes': False, 'oracle_types': False }, ({}, {}), None, fold_number))
    for _, predictor in fold_parameters_provider.get_fold_text_attributes_predictors(fold_number).items():
        run_parameters_combinations.append((fold_test_data,{ 'oracle_sentences': False, 'oracle_attributes': False, 'oracle_types': False }, ({}, {}), predictor, fold_number))

response_model_cache: dict[str, dict[int, type[BaseModel]]] = {}

for model_name in MODELS_TO_TEST:
    for test_data, experiment_parameters, (attr_titles, attr_descriptions), text_attributes_predictor, fold_number in run_parameters_combinations:
        
        experiment_runner = LLMOracleExperimentRunner(
            response_model_cache=response_model_cache,
            dataset_version=used_version,
            fold_number=fold_number,
            model_name=model_name,
            used_attributes=ALL_SELECTED_ATTRIBUTES_V2,
            attribute_types_dict=attributes_types_dict,
            attribute_titles_dict=attr_titles,
            attribute_descriptions_dict=attr_descriptions,
            text_attributes_predictor=text_attributes_predictor,
            **experiment_parameters,
            response_model_name="ApartmentDetails",
            temperature=TEMPERATURE,
            max_retries=MAX_RETRIES,
            max_tokens=MAX_TOKENS,
            seed=SEED)

        result_file = f"./results/{experiment_runner.model_identifier_string}.json"
        result_dict_list: list[dict] = load_experiment_data(result_file)

        already_processed_ids = {result[SampleProcessingResult.ID] for result in result_dict_list}
        data_to_process = [sample for sample in test_data if sample.id not in already_processed_ids]

        if len(data_to_process) == 0:
            print(f"{experiment_runner.model_identifier_string} already done.")
            continue

        print(f"Running {experiment_runner.model_identifier_string}...")
        print(f"Number of samples to process: {len(data_to_process)}")

        checkpoint_to_store = int(len(data_to_process) * 0.10)

        result_dictionary = experiment_runner.configuration | {
            "results": result_dict_list
        }
        
        is_working = experiment_runner.warmup_model()
    
        for result in experiment_runner.run_experiment(data_to_process):
            result_dict_list.append(result.to_dict())
            if len(result_dict_list) % STORE_AFTER == 0:
                store_experiment_data(result_file, result_dictionary)
            if len(result_dict_list) % checkpoint_to_store == 0:
                done_perc = round((len(result_dict_list) / len(test_data)) * 100)
                print(f"{experiment_runner.model_identifier_string}: {done_perc}%")
        store_experiment_data(result_file, result_dictionary)