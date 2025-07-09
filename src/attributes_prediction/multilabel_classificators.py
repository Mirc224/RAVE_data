import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb

class ClassifierBase(ABC):
    def __init__(self, name: str):
        super().__init__()
        self._classifier_name = name

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        pass
    
    @property
    def classifier_name(self) -> str:
        return self._classifier_name

class MultiLabelClassifierBase(ClassifierBase):
    def __init__(self, name: str):
        super().__init__(name)
        self.models_dict: dict[str, BaseEstimator] = {}
        self.number_of_classes = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        self.models_dict.clear()
        self.number_of_classes = y_train.shape[1]
        for i, attribute in enumerate(range(self.number_of_classes)):
            X_train_transformed, y_train_transformed = self._transform_data(X_train, y_train[:,i])
            model = self._create_model(X_train_transformed, y_train_transformed, **kwargs)
            self.models_dict[attribute] = model

    def _transform_data(self, X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return X_train, y_train

    @abstractmethod
    def _create_model(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> BaseEstimator:
        pass

    @abstractmethod
    def _predict(self, model, X_test: np.ndarray) -> np.ndarray:
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        predictions = np.zeros((X_test.shape[0], self.number_of_classes))
        for i, attribute in enumerate(range(self.number_of_classes)):
            predictions[:, i] = self._predict(self.models_dict[attribute], X_test)
        return predictions
    
class LogisticRegressionMultiClassifier(MultiLabelClassifierBase):
    def __init__(self):
        super().__init__("LogReg")

    def _create_model(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> BaseEstimator:
        model = LogisticRegression(**kwargs)
        model.fit(X_train, y_train)
        return model
    
    def _predict(self, model: LogisticRegression, X_test: np.ndarray) -> np.ndarray:
        return model.predict(X_test)

class SVMMultiClassifier(MultiLabelClassifierBase):
    def __init__(self):
        super().__init__("SVM")

    def _create_model(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> BaseEstimator:
        model = SVC(**kwargs)
        model.fit(X_train, y_train)
        return model
    
    def _predict(self, model: SVC, X_test: np.ndarray) -> np.ndarray:
        return model.predict(X_test)
    

class XGBoostMultiClassifier(MultiLabelClassifierBase):
    def __init__(self):
        super().__init__("XGBoost")

    def _create_model(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> BaseEstimator:
        model = XGBClassifier(**kwargs)
        model.fit(X_train, y_train)
        return model
    
    def _predict(self, model: XGBClassifier, X_test: np.ndarray) -> np.ndarray:
        return model.predict(X_test)
    
class LightGBMMultiClassifier(MultiLabelClassifierBase):
    def __init__(self):
        super().__init__("LightGBM")

    def _create_model(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> BaseEstimator:
        model = lgb.LGBMClassifier(**kwargs)
        model.fit(X_train, y_train)
        return model
    
    def _predict(self, model: XGBClassifier, X_test: np.ndarray) -> np.ndarray:
        return model.predict(X_test)
    
class KNNClassifierBase(MultiLabelClassifierBase):
    def __init__(self, n_estimators: int):
        super().__init__(f"{n_estimators}-NN")
        self.model: KNeighborsClassifier = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        X_train_transformed, y_train_transformed = self._transform_data(X_train, y_train)
        self.model = self._create_model(X_train_transformed, y_train_transformed, **kwargs)

    def _transform_data(self, X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return X_train, y_train

    def _create_model(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> BaseEstimator:
        model = KNeighborsClassifier(**kwargs)
        model.fit(X_train, y_train)
        return model

    def _predict(self, model, X_test: np.ndarray) -> np.ndarray:
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)