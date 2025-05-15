
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable
import instructor
from openai import OpenAI
from pydantic import BaseModel, create_model
from rave_dataset import RaveSample, RaveSentence
from openai.types.chat.chat_completion import ChatCompletion
import time
from response_model_creation import FieldBluePrint, ModelBluePrint
from utility import create_response_model
import datetime
from text_attributes_predictor import TextAttributesPredictor

class SampleProcessingResult:
    ID = "id"
    RESULT_RESPONSES = "responses"
    MODEL_RESPONSE = "response"
    USED_PROMPT_TOKENS = "usedPromptTokens"
    USED_COMPLETION_TOKENS = "usedCompletionTokens"
    TOTAL_USED_TOKENS = "totalUsedTokens"
    START_TIME = "startTime"
    END_TIME = "endTime"
    CREATED = "created"
    def __init__(self, id: int):
        self.__id = id
        self.__completion_responses: list[dict] = []
        self.__start_time: float = 0
        self.__end_time: float = 0
        self.__created: str = datetime.datetime.now().isoformat()
    
    @property
    def id(self) -> int:
        return self.__id
    
    @property
    def start_time(self) -> float:
        return self.__start_time
    
    @start_time.setter
    def start_time(self, value: float):
        self.__start_time = value

    @property
    def end_time(self) -> float:
        return self.__end_time
    
    @end_time.setter
    def end_time(self, value: float):
        self.__end_time = value
    
    @property
    def created(self) -> str:
        return self.__created
    
    @created.setter
    def created(self, value: str):
        self.__created = value

    @property
    def completion_responses(self) -> list[dict]:
        return self.__completion_responses
    
    @staticmethod
    def from_dict(data: dict) -> SampleProcessingResult:
        result = SampleProcessingResult(data[SampleProcessingResult.ID])
        for response in data[SampleProcessingResult.RESULT_RESPONSES]:
            result.__completion_responses.append(response)
        result.start_time = data[SampleProcessingResult.START_TIME]
        result.end_time = data[SampleProcessingResult.END_TIME]
        result.created = data[SampleProcessingResult.CREATED]
        return result
    
    def parse_completion_response(self, completion: ChatCompletion) -> dict:
        return {
            self.MODEL_RESPONSE : completion.choices[0].message.content,
            self.USED_PROMPT_TOKENS : completion.usage.prompt_tokens,
            self.USED_COMPLETION_TOKENS : completion.usage.completion_tokens,
            self.TOTAL_USED_TOKENS : completion.usage.total_tokens
        }

    def add_completion_response(self, completion: ChatCompletion):
        parsed_completion = self.parse_completion_response(completion)
        self.__completion_responses.append(parsed_completion)

    def to_dict(self) -> dict:
        return {
            self.ID : self.id,
            self.RESULT_RESPONSES : self.completion_responses,
            self.START_TIME: self.start_time,
            self.END_TIME: self.end_time,
            self.CREATED: self.created
        }


class LLMExperimentRunnerBase(ABC):
    INFO_DELIMITER = "_"
    def __init__(
            self,
            dataset_version: str,
            fold_number: str,
            model_name: str, 
            used_attributes: list[str],
            attribute_types_dict: dict[str, list[type]],
            attribute_titles_dict: dict[str, str] = {},
            attribute_descriptions_dict: dict[str, str] = {},
            **kwargs: dict):
        self.__dataset_version = dataset_version
        self.__fold_number = fold_number
        self.__sentence_delimiter = " "
        self.__model_kwargs = kwargs
        self.__model_name = model_name
        self.__client: instructor.client.Instructor = self.__init_client()
        self._attribute_names_mapping: dict[str, str] = {}
        self._attribute_titles: dict[str, str] = attribute_titles_dict
        self._attribute_descriptions: dict[str, str] = attribute_descriptions_dict
        self._base_messages: list[dict] = []
        self._used_attributes: set[str] = set(used_attributes)
        self._attribute_types_dict: dict[str, list[type]] = attribute_types_dict

    @property
    def attributes_names_mapping(self) -> dict[str, str]:
        return self._attribute_names_mapping
    
    @attributes_names_mapping.setter
    def attributes_names_mapping(self, value: dict[str, str]):
        self._attribute_names_mapping = value

    @property
    def attribute_titles(self) -> dict[str, str]:
        return self._attribute_titles
    
    @attribute_titles.setter
    def attribute_titles(self, value: dict[str, str]):
        self._attribute_titles = value

    @property
    def attribute_description(self) -> dict[str, str]:
        return self._attribute_descriptions
    
    @attribute_description.setter
    def attribute_description(self, value: dict[str, str]):
        self._attribute_descriptions = value
        
    def __init_client(self) -> instructor.client.Instructor:
        openai_instance = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        return instructor.from_openai(openai_instance, mode=instructor.Mode.JSON)
    
    def warmup_model(self) -> bool:
        return self.__client.chat.completions.create(
            model=self.__model_name,
            messages=[
                {
                    "role": "user",
                    "content": "Hello world!",
                }
            ],
            response_model=None
        ) is not None

    def run_experiment(self, data_samples: list[RaveSample], localization: str="en") -> Iterable[SampleProcessingResult]:
        for sample in data_samples:
            yield self.handle_sample(sample, self.get_response_model(sample, localization), localization)
    
    def handle_sample(self, sample: RaveSample, response_model_type: type, localization: str="en") -> SampleProcessingResult:
        result_object = SampleProcessingResult(sample.id)
        self.__client.hooks.clear()
        self.__client.on("completion:response", lambda response: result_object.add_completion_response(response))
        messages: list[dict] = []
        messages.extend(self._base_messages)
        messages.append(
            {
                "role": "user",
                "content": self.get_sample_text(sample, localization),
            }
        )
        try:
            result_object.start_time = time.perf_counter()
            self.__client.chat.completions.create(
                model=self.__model_name,
                messages=messages,
                response_model=response_model_type,
                **self.__model_kwargs
            )
        except:
            pass
        result_object.end_time = time.perf_counter()
        return result_object

    @abstractmethod
    def get_response_model(self, sample: RaveSample, localization: str="en") -> type[BaseModel]:
        pass
    
    def get_localized_text(self, list_of_sentences: list[RaveSentence], localization: str) -> str:
        return self.__sentence_delimiter.join([sentence.text(localization) for sentence in list_of_sentences])

    def get_sample_text(self, sample: RaveSample, localization: str) -> type[BaseModel]:
        return self.get_localized_text(sample.sentences, localization)

    @property
    def configuration(self) -> dict:
        return {
            "runnerName" : self.__class__.__name__,
            "completionParameters": self.__model_kwargs,
            "modelName": self.__model_name,
            "attributeNamesMapping": self._attribute_names_mapping,
            "attributeTitles": self._attribute_titles,
            "attributeDescriptions": self._attribute_descriptions,
            "sentenceDelimiter": self.__sentence_delimiter,
            "baseMessages": self._base_messages,
            "usedAttributes": list(self._used_attributes)
        }
    
    def _create_model_blueprint(self, response_model_name: str, used_attributes: Iterable[str]) -> ModelBluePrint:
        result_model_definition = ModelBluePrint(response_model_name)
        for attr_name in used_attributes:
            attr_field = FieldBluePrint(
                self._attribute_names_mapping.get(attr_name, attr_name), 
                self._attribute_types_dict[attr_name],
                self._attribute_titles.get(attr_name, None),
                self._attribute_descriptions.get(attr_name, None))
            result_model_definition.add_field(attr_field)
        return result_model_definition
    
    @property
    def experiment_identifier_string(self) -> str:
        additional_info: list[str] = [self.__dataset_version, self.__fold_number]
        if self._attribute_names_mapping:
            additional_info.append("am")
        if self._attribute_titles:
            additional_info.append("at")
        if self._attribute_descriptions:
            additional_info.append("ad")
        return self.INFO_DELIMITER.join(additional_info)
    
    @property
    def model_identifier_string(self) -> str:
        return self.INFO_DELIMITER.join([self.__model_name.replace(":","_"), self.experiment_identifier_string])
    
class LLMOracleExperimentRunner(LLMExperimentRunnerBase):
    def __init__(
            self, 
            response_model_cache: dict[str, dict[int, type[BaseModel]]],
            dataset_version: str,
            fold_number: str,
            model_name: str, 
            used_attributes: list[str], 
            attribute_types_dict: dict[str, list[type]],
            attribute_titles_dict: dict[str, str] = {},
            attribute_descriptions_dict: dict[str, str] = {},
            oracle_attributes: bool = True,
            oracle_types: bool = True,
            oracle_sentences: bool = True,
            text_attributes_predictor: TextAttributesPredictor = None,
            response_model_name: str = "ApartmentDetails",
            **kwargs):
        super().__init__(dataset_version, fold_number, model_name, used_attributes, attribute_types_dict, attribute_titles_dict, attribute_descriptions_dict, **kwargs)
        self.__text_atributes_predictor: TextAttributesPredictor = text_attributes_predictor
        self.__oracle_attributes = oracle_attributes
        self.__oracle_types = oracle_types
        self.__oracle_sentences = oracle_sentences
        self.__response_model_name: str = response_model_name
        self.__default_blueprint: ModelBluePrint = self._create_model_blueprint(self.__response_model_name, self._used_attributes)
        self.__default_type: type[BaseModel] = create_response_model(self.__default_blueprint)
        
        experiment_identifier = self.experiment_identifier_string
        self.__experiment_cache = response_model_cache.get(experiment_identifier, {})
        response_model_cache[experiment_identifier] = self.__experiment_cache

    
    def get_response_model(self, sample: RaveSample, localization: str="en") -> type[BaseModel]:
        if not self.__oracle_attributes and not self.__oracle_types and self.__text_atributes_predictor is None:
            return self.__default_type
        
        sample_response_model = self.__experiment_cache.get(sample.id)
        if sample_response_model is not None:
            return sample_response_model

        model_attributes: set[str] = self._handle_used_attributes(sample, localization) & self._used_attributes
        model_blueprint = self._create_model_blueprint(self.__response_model_name, model_attributes)
        
        if not self.__oracle_types:
            sample_response_model = create_response_model(model_blueprint)
            self.__experiment_cache[sample.id] = sample_response_model
            return sample_response_model
        
        for attr_name in model_attributes:
            model_blueprint.get_field(attr_name).field_type = type(sample.text_attributes[attr_name])

        sample_response_model = create_response_model(model_blueprint)
        self.__experiment_cache[sample.id] = sample_response_model
        return sample_response_model
    
    def get_sample_text(self, sample: RaveSample, localization: str) -> type[BaseModel]:
        if not self.__oracle_sentences:
            return super().get_sample_text(sample, localization)
        filtred_sentences = sample.get_sentences_with_attributes(sample.text_attributes.keys())
        return self.get_localized_text(filtred_sentences, localization)
    
    def _handle_used_attributes(self, sample: RaveSample, localization: str="en") -> set[str]:
        if self.__oracle_attributes and self.__text_atributes_predictor is None:
            return self._used_attributes & set(sample.text_attributes.keys())
        
        if self.__text_atributes_predictor is not None:
            return self.__predict_sample_attributes(sample, localization)

        return self._used_attributes
    
    def __predict_sample_attributes(self, sample: RaveSample, localization: str="en") -> set[str]:
        return set.union(*self.__text_atributes_predictor.predict(sample.text(localization)))
    
    @property
    def experiment_identifier_string(self) -> str:
        additional_info: list[str] = [super().experiment_identifier_string]
        if self.__oracle_attributes:
            additional_info.append("oa")
        if self.__oracle_types:
            additional_info.append("ot")
        if self.__oracle_sentences:
            additional_info.append("os")
        if self.__text_atributes_predictor is not None:
            additional_info.append(self.__text_atributes_predictor.description)
        return self.INFO_DELIMITER.join(additional_info)
