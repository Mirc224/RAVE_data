import numpy as np
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer

class VectorizerBase(ABC):
    def __init__(self):
        super().__init__()
        self._name = "vectorizerBase"

    @abstractmethod   
    def new_vectorizer(self) :
        pass

    @abstractmethod
    def train(self, sentences: list[str]):
        pass

    @abstractmethod
    def vectorize(self, sentences: list[str]) -> np.ndarray:
        pass

    @property
    def vectorizer_name(self) -> str:
        return self._name

class BoWVectorizer(VectorizerBase):
    def __init__(self):
        super().__init__()
        self.__vectorizer = self.new_vectorizer()
        self._name = "bow"

    def new_vectorizer(self) -> CountVectorizer:
        return CountVectorizer(max_features=5000, stop_words="english")

    def train(self, sentences: list[str]):
        self.__vectorizer = self.new_vectorizer()
        self.__vectorizer.fit(sentences)
    
    def vectorize(self, sentences: list[str]) -> np.ndarray:
        return self.__vectorizer.transform(sentences).toarray()
    
    
class TFIDFVectorizer(VectorizerBase):
    def __init__(self):
        super().__init__()
        self.__vectorizer = self.new_vectorizer()
        self._name = "tfidf"

    def new_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(max_features=5000, stop_words="english")

    def train(self, sentences: list[str]):
        self.__vectorizer = self.new_vectorizer()
        self.__vectorizer.fit(sentences)
    
    def vectorize(self, sentences: list[str]) -> np.ndarray:
        return self.__vectorizer.transform(sentences).toarray()
    
class EmbeddingsVectorizer(VectorizerBase):
    def __init__(self):
        super().__init__()
        self.__vectorizer = self.new_vectorizer()
        self._name = "embeddings"

    def new_vectorizer(self) -> SentenceTransformer:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def train(self, sentences: list[str]):
        pass
    
    def vectorize(self, sentences: list[str]) -> np.ndarray:
        return self.__vectorizer.encode(sentences)