from pydantic import BaseModel, create_model
from models.response_model_creation import ModelBluePrint
from models.text_samples import TextSample, Sentence
from typing import Generator
from pathlib import Path
from models.rave_dataset import *
import json
import itertools
import ollama
import typing
from collections import Counter
import os

def list_iterator_func(lines: list[str]):
    for line in lines:
        yield line

def convert_to_best_type(value: str):
    value = value.strip()
    value_lower = value.lower()
    if value_lower == "true":
        return True
    elif value_lower == "false":
        return False

    try:
        float_value = float(value)
        if float_value.is_integer():
            return int(float_value)
        if "." in value or "e" in value.lower():  # Looks for decimal point or scientific notation
            return float_value
    except ValueError:
        pass
    return value

def read_raw_sentences(iterator: Generator[str, any, None], end_terminator: str) -> list[str]:
    result_list = []
    for line in iterator:
        if line == end_terminator:
            break
        result_list.append(line)
    return result_list

def read_raw_attributes(iterator: Generator[str, any, None], end_terminator: str, split_symbol="|") -> dict[str,int|float|str|bool]:
    result_dict:dict[str,int|float|str] = {}
    for line in iterator:
        if line == end_terminator:
            break
        splitted_attribute = [item.strip() for item in line.split(split_symbol)]
        key, value = splitted_attribute[0], convert_to_best_type(splitted_attribute[1])
        result_dict[key] = value
    return result_dict

def parse_sentence_and_attributes(raw_sentence: str, attribute_separator:str =" @") -> tuple[str, set[str]]:
    splitted_sentence = raw_sentence.split(attribute_separator)
    return splitted_sentence[0].strip(), {item.strip() for item in splitted_sentence[1:] }

def read_sample_lines(iterator: Generator[str, any, None], end_terminator: str = "}"):
    sample_lines: list[str] = []
    for line in iterator:
        line = line.strip()
        if not line:
            continue
        if line == end_terminator:
            break
        sample_lines.append(line)
    return sample_lines

def parse_text_sample(sample_lines: list[str], section_delimiter:str ="###", end_terminator: str = "}", attribute_delimiter: str= " @") -> TextSample:
    sample_lines_iterator = list_iterator_func(sample_lines)
    new_text_sample = TextSample()
    attributes = read_raw_attributes(sample_lines_iterator, section_delimiter)
    for key, value in attributes.items():
        new_text_sample.add_attribute(key, value)
    sample_sentences = read_raw_sentences(sample_lines_iterator, end_terminator)
    for raw_sentence in sample_sentences:
        text, sentence_attributes = parse_sentence_and_attributes(raw_sentence, attribute_delimiter)
        sentence = Sentence(text, sentence_attributes)
        new_text_sample.add_sentence(sentence)  
    return new_text_sample


def read_raw_data(file_path: str) -> list[TextSample]:
    raw_data_path = Path(file_path)
    with open(raw_data_path, "r", encoding="utf8") as f:
        loaded_lines: list[str] = [line.strip() for line in f.readlines() if line.strip()]

    result_samples_list: list[TextSample] = []
    i = 0
    list_iterator = list_iterator_func(loaded_lines)
    for line in list_iterator:
        line = line.strip()
        if line != "{":
            continue
        i += 1
        sample_lines = read_sample_lines(list_iterator, "}")
        new_text_sample = parse_text_sample(sample_lines)
        result_samples_list.append(new_text_sample)
    return result_samples_list

def load_rave_dataset(dataset_path: str) -> list[RaveSample]:
    with open(dataset_path, "r", encoding="utf8") as f:
        data = json.load(f)
    return [RaveSample(raw_data_sample) for raw_data_sample in data]

def create_response_model(model_blueprint: ModelBluePrint) -> type[BaseModel]:
    model_fields_dict: dict = {}
    for field in model_blueprint.fields:
        model_fields_dict[field.field_name] = field.field_definition_tuple
    return create_model(model_blueprint.model_name, **model_fields_dict)

def get_experiment_parameters(loaded_attribute_descriptions: dict[str, str]) -> list[tuple[dict, tuple[dict, dict]]]:
    oracle_keys = ["oracle_sentences", "oracle_attributes"]
    combinations = list(itertools.product([True, False], repeat=len(oracle_keys)))
    oracle_parameters_combinations = [dict(zip(oracle_keys, combination)) for combination in combinations]
    for oracle_parameters in oracle_parameters_combinations:
        oracle_parameters["oracle_types"] = oracle_parameters["oracle_attributes"]
    description_combinations = list(itertools.product([{}], [{}, loaded_attribute_descriptions]))
    return list(itertools.product(oracle_parameters_combinations, description_combinations))

def pull_ollama_models(required_models: list[str]):
    current_ollama_models: set[str] = {model_desc.model for model_desc in ollama.list().models}
    for model_to_check in required_models:
        if model_to_check in current_ollama_models:
            continue
        ollama.pull(model_to_check)

def get_attribute_types_according_to_data(dataset: list[RaveSample]) -> dict[str, list[type]]:
    all_attributes_types_list: dict[str, list[type]] = {}
    for sample in dataset:
        for attribute_name in sample.text_attributes.keys():
            attribute_type_list = all_attributes_types_list.get(attribute_name, [])
            attribute_type_list.append(type(sample.text_attributes[attribute_name]))
            all_attributes_types_list[attribute_name] = attribute_type_list

    attributes_types_dict: dict[str, list[type]] = {}
    for attribute_name, types_count_list in all_attributes_types_list.items():
        dynamic_union = typing.Union[None]
        for attr_type, _ in Counter(types_count_list).most_common():
            dynamic_union = typing.Union[attr_type, dynamic_union]
        attributes_types_dict[attribute_name] = dynamic_union
    return attributes_types_dict

def load_experiment_data(file_path: str) -> list[dict]:
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r', encoding="utf8") as file:
        data = json.load(file)["results"]
    return data

def store_experiment_data(file_path: str, results: dict[dict]):
    with open(file_path, "w+", encoding="utf8") as f:
        json.dump(results, f)