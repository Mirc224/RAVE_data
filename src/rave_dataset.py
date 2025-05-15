from rave_constants import *
class RaveSentence:
    def __init__(self, order_number: int, sentence_dict: str):
        self.__text: dict = self.__handle_sentnece_dict(sentence_dict)
        self.__attributes = set(sentence_dict[SENTENCE_ATTRIBUTES_KEY])
        self.__order_number = order_number

    def text(self, localization: str) -> str:
        return self.__text[localization]
    
    def has_attribute(self, attribute: str) -> bool:
        return attribute in self.attributes
    
    def has_any_of_attributes(self, attributes: set) -> bool:
        return len(attributes & self.attributes) > 0
    
    def __handle_sentnece_dict(self, sentence_dict: dict) -> dict[str, str]:
        excluded_keys = [SENTENCE_ATTRIBUTES_KEY]
        result_text_dict: dict[str, str] = {}
        for localization in {key for key in sentence_dict.keys() if key not in excluded_keys}:
            result_text_dict[localization] = sentence_dict[localization]
        return result_text_dict
    
    def to_dict(self) -> dict:
        return {
            SENTENCE_ATTRIBUTES_KEY: list(self.__attributes),
        } | self.all_text_dict
    
    @property
    def all_text_dict(self) -> dict[str, str]:
        return self.__text

    @property
    def attributes(self) -> set:
        return self.__attributes
    
    @property
    def order_number(self) -> int:
        return self.__order_number

class RaveSample:
    def __init__(self, data_sample: dict):
        self.__id = data_sample[SAMPLE_ID_KEY]
        self.__text_attributes: dict = dict(data_sample[TEXT_ATTRIBUTES_KEY])
        self.__sentences: list[RaveSentence]  = self.__handle_sentences(data_sample)

    def __handle_sentences(self, data_sample: dict) -> list[RaveSentence]:
        result: list[RaveSentence] = []
        for sentence_order, sentence_item in enumerate(data_sample[SAMPLE_SENTENCES_KEY]):
            result.append(RaveSentence(sentence_order, sentence_item))
        return result
    
    def text(self, localization: str) -> list[str]:
        return [sentence.text(localization) for sentence in self.sentences]
    
    def get_sentences_with_attributes(self, required_attributes: list[str]) -> list[RaveSentence]:
        required_attributes_set = set(required_attributes)
        return [sentence for sentence in self.sentences if sentence.has_any_of_attributes(required_attributes_set)]
    
    def to_dict(self) -> dict:
        return {
            SAMPLE_ID_KEY: self.id,
            TEXT_ATTRIBUTES_KEY: self.text_attributes,
            SAMPLE_SENTENCES_KEY: [sentence.to_dict() for sentence in self.sentences]
        }

    def keep_selected_text_attributes(self, selected_attributes: list):
        filtred_attributes = {}
        for key, value in self.text_attributes.items():
            if key not in selected_attributes:
                continue
            filtred_attributes[key] = value
        self.__text_attributes = filtred_attributes

    @property
    def text_attributes(self) -> dict[str, str|int|float]:
        return self.__text_attributes
    
    @property
    def all_sentence_attributes(self) -> set[str]:
        result: list[str] = []
        for sentence in self.sentences:
            result.extend(sentence.attributes)
        return set(result)

    @property
    def sentences(self) -> list[RaveSentence]:
        return self.__sentences
    
    @property
    def id(self) -> int:
        return self.__id