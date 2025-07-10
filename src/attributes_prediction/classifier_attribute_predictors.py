from typing import Generator, Any
import numpy as np
from models.rave_dataset import RaveSample
from vectorization.vectorizer_utils import create_binary_vector
from vectorization.vectorizers import VectorizerBase
from attributes_prediction.multilabel_classificators import ClassifierBase
from sklearn.base import BaseEstimator

class ClassifierAttributePredictor:
    def __init__(self, vectorizer: VectorizerBase, classifier: ClassifierBase, ordered_attributes: list[str]):
        self._ordered_attributes = ordered_attributes
        self._vectorizer = vectorizer
        self._classifier = classifier
        self._models_dict: dict[str, BaseEstimator] = {}

    def fit(self, train_samples: list[RaveSample], train_params: dict):
        sentences = [sentence.text("en") for sample in train_samples for sentence in sample.sentences]
        vectorized_sentences = self._vectorizer.vectorize(sentences)
        sentence_labels = [sentence.attributes for sample in train_samples for sentence in sample.sentences]
        vectorized_labels = np.array([create_binary_vector(self._ordered_attributes, labels) for labels in sentence_labels])
        self._classifier.fit(vectorized_sentences, vectorized_labels, **train_params)

    def predict(self, test_samples: list[RaveSample]) -> Generator[list[str], Any, None]:
        for sample in test_samples:
            vectorized_sentences = self._vectorizer.vectorize([sentence.text("en") for sentence in sample.sentences])
            predicted_vectors = self._classifier.predict(vectorized_sentences)
            yield self._get_predicted_lables(predicted_vectors)
    
    def _get_predicted_lables(self, predicted: np.ndarray) -> list[list[str]]:
        result: list[list[str]] = []
        for predicted_labels in predicted:
            result.append([value for i, value in enumerate(self._ordered_attributes) if predicted_labels[i] == 1])
        return list(set().union(*result))
    
    @property
    def description(self) -> str:
        return f"{self._vectorizer.vectorizer_name}_{self._classifier.classifier_name}"