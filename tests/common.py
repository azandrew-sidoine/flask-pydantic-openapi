from datetime import date
from enum import IntEnum, Enum
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field, model_validator


class Order(IntEnum):
    asce = 1
    desc = 0


class Query(BaseModel):
    order: Annotated[Order, Optional] | None = Field(None, validate_default=True)


class QueryParams(BaseModel):
    name: Optional[List[str]]
    columns: Optional[List[str]]


class User(BaseModel):
    name: str


class Users(BaseModel):
    data: List[User]
    columns: List[str]


class JSON(BaseModel):
    name: str
    limit: int


class Resp(BaseModel):
    name: str
    score: List[int]


class Language(str, Enum):
    en = "en-US"
    zh = "zh-CN"


class Headers(BaseModel):
    lang: Language

    @model_validator(mode='before')
    def lower_keys(cls, values):
        return {key.lower(): value for key, value in values.items()}


class Cookies(BaseModel):
    pub: List[str]


class DemoModel(BaseModel):
    uid: int
    limit: int
    name: str


class FileMetadata(BaseModel):
    type: str
    created_at: date


class FileName(BaseModel):
    file_name: str
    data: FileMetadata


def get_paths(spec):
    paths = []
    for path in spec["paths"]:
        if spec["paths"][path]:
            paths.append(path)

    paths.sort()
    return paths
