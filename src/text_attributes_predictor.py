import numpy as np
from vectorizers import VectorizerBase
from multilabel_classificators import ClassifierBase
from vectorizer_utils import create_binary_vector

class TextAttributesPredictor:
    def __init__(self, 
                 vectorizer: VectorizerBase, 
                 classifier: ClassifierBase, 
                 vectorizer_parameters: dict, 
                 classifier_parameters: dict, 
                 ordered_attributes: list[str] = None):
        self._vectorizer_parameters = vectorizer_parameters
        self._classifier_parameters = classifier_parameters
        self._vectorizer = vectorizer
        self._classifier = classifier
        self._ordered_attributes: list[str] = ordered_attributes if ordered_attributes is not None else []
    
    def train(self, sentences: list[str], sentences_labels: list[set[str]], ordered_attributes: list[str]):
        self._ordered_attributes = ordered_attributes

        vectorized_labels = np.array([create_binary_vector(self._ordered_attributes, labels) for labels in sentences_labels])
        self._vectorizer.train(sentences)
        vectorized_sentences = self._vectorizer.vectorize(sentences)
        self._classifier.fit(vectorized_sentences, vectorized_labels, **self._classifier_parameters)

    def predict(self, sentences: list[str]) -> list[set[str]]:
        vectorized_sentences = self._vectorizer.vectorize(sentences)
        predicted = self._classifier.predict(vectorized_sentences)
        return self._get_predicted_labels(predicted)
    
    def _get_predicted_labels(self, predicted: np.ndarray) -> list[set[str]]:
        result: list[list[str]] = []
        for predicted_labels in predicted:
            result.append(set([value for i, value in enumerate(self._ordered_attributes) if predicted_labels[i] == 1]))
        return result
    
    @property
    def description(self) -> str:
        return f"{self._vectorizer.vectorizer_name}_{self._classifier.classifier_name}"