class Sentence:
    def __init__(self, text: str, attributes: set[str]):
        self.__text: str = text
        self.__attributes: set[str] = attributes

    @property
    def text(self) -> str:
        return self.__text
    
    @property
    def attributes(self) -> set[str]:
        return self.__attributes
    
class TextSample:
    def __init__(self):
        self.__attributes: dict[str, float|int|bool|str] = {}
        self.__sentences: list[Sentence] = []

    @property
    def sentences(self) -> list[Sentence]:
        return self.__sentences
    
    @property
    def attributes(self) -> dict[str, float|int|bool|str]:
        return self.__attributes
    
    def add_sentence(self, sentence: Sentence):
        self.__sentences.append(sentence)

    def add_attribute(self, key: str, value: int|bool|float|str):
        self.__attributes[key] = value
    
    @property
    def text(self) -> str:
        return "\n".join([sentence.text for sentence in self.__sentences])