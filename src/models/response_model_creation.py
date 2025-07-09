from pydantic import Field
from pydantic.fields import FieldInfo
from typing import Any
from typing import Iterable

class FieldBluePrint:
    def __init__(self, 
                 field_name: str, 
                 field_type: type,
                 title:str|None = None, 
                 description: str|None = None):
        self.__field_name = field_name
        self.__field_type = field_type
        field_annotations = {
        }
        if title:
            field_annotations["title"] = title
        if description:
            field_annotations["description"] = description
        self.__field_annotation = Field(**field_annotations) if field_annotations else None

    @property
    def field_name(self) -> str:
        return self.__field_name
    
    @property
    def field_type(self) -> type:
        return self.__field_type
    
    @field_type.setter
    def field_type(self, value: type):
        self.__field_type = value
    
    @property
    def field_annotation(self) -> FieldInfo|None:
        return self.__field_annotation
    
    @property
    def field_definition_tuple(self) -> tuple:
        if self.field_annotation:
            return (self.field_type, self.field_annotation)
        return (self.field_type, ...)

class ModelBluePrint:
    def __init__(self, name: str):
        self.__name = name
        self.__fields: dict[str, FieldBluePrint] = {}
    
    def add_field(self, new_field: FieldBluePrint):
        self.__fields[new_field.field_name] = new_field
    
    def add_fields(self, new_fields: list[FieldBluePrint]):
        for field in new_fields:
            self.add_field(field)

    def get_field(self, field_name: str) -> FieldBluePrint:
        return self.__fields[field_name]

    @property
    def model_name(self) -> str:
        return self.__name
    
    @property
    def fields(self) -> Iterable[FieldBluePrint]:
        return self.__fields.values()
