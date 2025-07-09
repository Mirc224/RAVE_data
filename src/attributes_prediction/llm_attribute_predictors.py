
import instructor
import copy
import random
import time
from openai import OpenAI
from pydantic import BaseModel
from typing import Any, Generator, Iterable
from models.rave_dataset import RaveSample, RaveSentence
from models.response_models import PresentAttributes
from experiments.running_experiments import SampleProcessingResult

ROLE_KEY = "role"
CONTENT_KEY = "content"
USER_KEY = "user"
SYSTEM_KEY = "system"
ASSISTANT_KEY = "assistant"

class ExampleSelectorBase:
    def __init__(self, samples: list[RaveSample], seed:int=12345):
        self.__seed = seed
        self.__random = random.Random(self.__seed)
        self.__samples: list[RaveSample] = samples

    def reset(self):
        self.__random = random.Random(self.__seed)

    def select_n_examples(self, number_of_examples: int, input_sample: RaveSample, localization: str) -> Generator[RaveSample, Any, None]:
        if number_of_examples < 1:
            return
        
        for example in self.__random.choices(self.__samples, k=number_of_examples):
            yield example
    
    @property
    def selector_identifier_string(self) -> str:
        return "_".join("random")

class LLMAttributePredictorRunnerBase:
    INFO_DELIMITER = "_"
    def __init__(
            self,
            dataset_version: str,
            fold_number: str,
            model_name: str,
            used_attributes: list[str],
            number_of_examples: int,
            example_selector: ExampleSelectorBase,
            **kwargs: dict):
        self.__dataset_version = dataset_version
        self.__fold_number = fold_number
        self.__sentence_delimiter = " "
        self.__model_kwargs = kwargs
        self.__model_name = model_name
        self.__number_of_examples = number_of_examples
        self.__example_selector = example_selector
        self.__example_selector.reset()
        self.__client: instructor.client.Instructor = self.__init_client()
        self._used_attributes: set[str] = set(used_attributes)
        self._base_messages: list[dict] = [
            {
                ROLE_KEY: SYSTEM_KEY,
                CONTENT_KEY: f"""
                        You are a classifier that identifies which of the following real estate attributes are mentioned in the text: 
                        {self.__format_example_output(self._used_attributes)}.

                        Return only the attribute names that are explicitly or implicitly mentioned.
                        Do not guess. Only return attributes that are clearly present.
                        """
            }
        ]

    def __init_client(self) -> instructor.client.Instructor:
        openai_instance = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        return instructor.from_openai(openai_instance, mode=instructor.Mode.JSON)
    
    def warmup_model(self) -> bool:
        return self.__client.chat.completions.create(
            model=self.__model_name,
            messages=[
                {
                    ROLE_KEY: USER_KEY,
                    CONTENT_KEY: "Hello world!",
                }
            ],
            response_model=None
        ) is not None

    def run_experiment(self, data_samples: list[RaveSample], localization: str="en") -> Iterable[SampleProcessingResult]:
        for sample in data_samples:
            yield self.handle_sample(sample, PresentAttributes, localization)

    def get_example_messages(self, input_sample: RaveSample, number_of_samples: int, localization: str) -> Generator[tuple[dict[str, str], dict[str, str]], Any, None]:
        for example in self.__example_selector.select_n_examples(number_of_samples, input_sample, localization):
            input_text = self.get_localized_text(example.sentences, localization)
            yield ({
                ROLE_KEY : USER_KEY,
                CONTENT_KEY : f"Input: {input_text}"
            }, 
            {
                ROLE_KEY : ASSISTANT_KEY,
                CONTENT_KEY : f'{{ "attributes": {self.__format_example_output(example.text_attributes.keys())} }}'
            })
    
    def handle_sample(self, sample: RaveSample, response_model_type: type, localization: str="en") -> SampleProcessingResult:
        result_object = SampleProcessingResult(sample.id)
        self.__client.hooks.clear()
        self.__client.on("completion:response", lambda response: result_object.add_completion_response(response))
        messages: list[dict] = []
        
        messages.extend([copy.deepcopy(message) for message in self._base_messages])
        input_examples: list[dict] = []
        for example_input, example_output in self.get_example_messages(sample, self.__number_of_examples, localization):
            messages.append(example_input)
            messages.append(example_output)
            input_examples.append(example_input)
            input_examples.append(example_output)

        result_object.examples = input_examples

        messages.append(
            {
                ROLE_KEY: USER_KEY,
                CONTENT_KEY: "Input: " + self.get_sample_text(sample, localization)
            })
        
        try:
            result_object.start_time = time.perf_counter()
            self.__client.chat.completions.create(
                model=self.__model_name,
                messages=messages,
                response_model=response_model_type,
                **self.__model_kwargs
            )
        except Exception as e:
            pass
        result_object.end_time = time.perf_counter()
        return result_object
    
    def get_localized_text(self, list_of_sentences: list[RaveSentence], localization: str) -> str:
        return self.__sentence_delimiter.join([sentence.text(localization) for sentence in list_of_sentences])

    def get_sample_text(self, sample: RaveSample, localization: str) -> type[BaseModel]:
        return self.get_localized_text(sample.sentences, localization)
    
    
    def __format_example_output(self, attributes: list[str]) -> str:
        expected_output = ', '.join([f'"{attribute_name}"' for attribute_name in attributes])
        return f'[{expected_output}]'

    @property
    def configuration(self) -> dict:
        return {
            "runnerName" : self.__class__.__name__,
            "completionParameters": self.__model_kwargs,
            "modelName": self.__model_name,
            "sentenceDelimiter": self.__sentence_delimiter,
            "baseMessages": self._base_messages,
            "usedAttributes": list(self._used_attributes)
        }
    
    @property
    def experiment_identifier_string(self) -> str:
        additional_info: list[str] = [self.__dataset_version, self.__fold_number, f'{self.__number_of_examples}shot']
        return self.INFO_DELIMITER.join(additional_info)
    
    @property
    def model_identifier_string(self) -> str:
        return self.INFO_DELIMITER.join([self.__model_name.replace(":","_"), self.experiment_identifier_string])