import os
import json
import numpy as np
from vectorizer_utils import create_binary_vector
from pathlib import Path
from rave_dataset import RaveSample
from text_attributes_predictor import TextAttributesPredictor
from vectorizers import VectorizerBase, BoWVectorizer, TFIDFVectorizer, EmbeddingsVectorizer
from multilabel_classificators import XGBoostMultiClassifier, LogisticRegressionMultiClassifier, SVMMultiClassifier, KNNClassifierBase, ClassifierBase

class FoldParametersProvider:
    def __init__(self):
        self._folds_category_ids: dict[str, dict[str, list[int]]] = {}
        self._folds_data: dict[str, dict[str, list[RaveSample]]] = {}
        self._text_attributes_predictors: dict[str, TextAttributesPredictor] = {}

    def read_folds_folder(self, folds_folder_path: Path) -> dict[str, dict[str, list[int]]]:
        folds_folder_path = Path("./dataset/folds/")
        result_folds_dict: dict[str, dict[str, list[int]]] = {}

        for file in os.listdir(folds_folder_path):
            fold_file = folds_folder_path / file
            fold_name = os.path.splitext(file)[0]
            splited_name = fold_name.split("_")
            fold_number = splited_name[2]
            fold_type = splited_name[3]

            fold_dict = result_folds_dict.get(fold_number, {})
            result_folds_dict[fold_number] = fold_dict
            with open(fold_file, "r", encoding="utf8") as f:
                fold_dict[fold_type] = json.load(f)
        return result_folds_dict
    
    def initialize_fold_data(self, rave_datase: list[RaveSample]):
        dataset_mapping = {sample.id:sample for sample in rave_datase}

        for fold_number, fold_data_dict in self._folds_category_ids.items():
            current_fold_data = self._folds_data.get(fold_number, {})
            self._folds_data[fold_number] = current_fold_data
            for fold_category, category_ids in fold_data_dict.items():
                current_fold_data[fold_category] = [dataset_mapping[sample_id] for sample_id in category_ids]

    def train_for_each_fold(self, ordered_attributes: list[str]):
        self._vectorizers = {}
        for fold_number, fold_data_category_dict in self._folds_data.items():
            fold_vectorizers = self._vectorizers.get(fold_number, {})
            self._vectorizers = fold_vectorizers
            fold_train_samples = fold_data_category_dict["train"]
            print(f"Training text predictors for fold: {fold_number}")
            self._text_attributes_predictors[fold_number] = self.__train_on_fold(fold_train_samples, ordered_attributes)

    def __train_on_fold(self, fold_train_samples: list[RaveSample], ordered_attributes: list[str]):
        X_data = [sentence.text("en") for sample in fold_train_samples for sentence in sample.sentences]
        Y_data = [sentence.attributes for sample in fold_train_samples for sentence in sample.sentences]
        vectorized_labels = np.array([create_binary_vector(ordered_attributes, labels) for labels in Y_data])
        
        vectorizers: list[VectorizerBase] = [
            BoWVectorizer(),
            TFIDFVectorizer(),
            EmbeddingsVectorizer()
        ]

        classifier: ClassifierBase
        train_params: dict
        result_text_attributes_predictors: dict[str, TextAttributesPredictor] = {}
        for vectorizer in vectorizers:
            vectorizer.train(X_data)
            classifiers = [
                (LogisticRegressionMultiClassifier(), {"max_iter" : 1000}),
                (SVMMultiClassifier(), {"kernel": "linear"}),
                (XGBoostMultiClassifier(), {"objective": "binary:logistic"}),
                (KNNClassifierBase(1), {"n_neighbors": 1}),
                (KNNClassifierBase(3), {"n_neighbors": 3}),
                (KNNClassifierBase(5), {"n_neighbors": 5}),
                (KNNClassifierBase(7), {"n_neighbors": 7}),
            ]

            for classifier, train_params in classifiers:
                predictor_name = f"{vectorizer.vectorizer_name}_{classifier.classifier_name}"
                print(f"Training {predictor_name}...")
                vectorized_sentences = vectorizer.vectorize(X_data)
                classifier.fit(vectorized_sentences, vectorized_labels, **train_params)
                result_text_attributes_predictors[predictor_name] = TextAttributesPredictor(
                    vectorizer, 
                    classifier, 
                    {},
                    train_params,
                    ordered_attributes)
        return result_text_attributes_predictors
    
    def get_fold_train_test_data(self, fold_number: str) -> tuple[list[RaveSample], list[RaveSample]]:
        fold_data = self._folds_data[fold_number]
        return fold_data["train"], fold_data["test"]
    
    def get_fold_text_attributes_predictors(self, fold_number: str) -> dict[str, TextAttributesPredictor]:
        return self._text_attributes_predictors[fold_number]

    @property
    def fold_numbers(self) -> list[str]:
        return list(self._folds_category_ids.keys())

    @property
    def folds_data(self) -> dict[str, dict[str, list[RaveSample]]]:
        return self._folds_data

    @property
    def folds_dict(self) -> dict[str, dict[str, list[int]]]:
        return self._folds_category_ids
    
    @folds_dict.setter
    def folds_dict(self, value: dict[str, dict[str, list[int]]]):
        self._folds_category_ids = value