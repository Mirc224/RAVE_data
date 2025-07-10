from typing import Generator, Any
from models.rave_dataset import RaveSample
from vectorization.vectorizers import BoWVectorizer, TFIDFVectorizer, EmbeddingsVectorizer, VectorizerBase
from attributes_prediction.multilabel_classificators import XGBoostMultiClassifier, LogisticRegressionMultiClassifier, SVMMultiClassifier, KNNClassifierBase
from attributes_prediction.text_attributes_predictor import TextAttributesPredictor
from attributes_prediction.classifier_attribute_predictors import ClassifierAttributePredictor


def train_attribute_predictors(
        train_data: list[RaveSample], 
        ordered_attributes: list[str], 
        vectorizer: VectorizerBase) -> Generator[TextAttributesPredictor, Any, None]:
    raw_classifiers = [
                (LogisticRegressionMultiClassifier(), {"max_iter" : 1000}),
                (SVMMultiClassifier(), {"kernel": "linear"}),
                (XGBoostMultiClassifier(), {"objective": "binary:logistic"}),
                (KNNClassifierBase(1), {"n_neighbors": 1}),
                (KNNClassifierBase(3), {"n_neighbors": 3}),
                (KNNClassifierBase(5), {"n_neighbors": 5}),
                (KNNClassifierBase(7), {"n_neighbors": 7})
            ]
    for classifier, train_params in raw_classifiers:
        predictor = ClassifierAttributePredictor(vectorizer, classifier, ordered_attributes)
        print(f"Training {predictor.description}...")
        predictor.fit(train_data, train_params)
        yield predictor


def train(train_data: list[RaveSample], ordered_attributes: list[str]) -> Generator[ClassifierAttributePredictor, Any, None]:
    vectorizers: list[VectorizerBase] = [
            BoWVectorizer(),
            TFIDFVectorizer(),
            EmbeddingsVectorizer()
    ]

    for vectorizer in vectorizers:
        X_sentences = [sentence.text("en") for sample in train_data for sentence in sample.sentences]
        vectorizer.train(X_sentences)
        for attribute_predictor in train_attribute_predictors(train_data, ordered_attributes, vectorizer):
            yield attribute_predictor