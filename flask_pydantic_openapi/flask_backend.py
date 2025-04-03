import gzip
import json
import logging

from typing import Optional, Mapping, Callable, Any, Tuple, List, Type, Iterable, Dict, Union
from dataclasses import dataclass

from pydantic import TypeAdapter, ValidationError, BaseModel
from flask import (
    request,
    abort,
    make_response,
    jsonify,
    Request as FlaskRequest,
    Flask,
    Response as FlaskResponse,
)
from werkzeug.datastructures import Headers
from werkzeug.routing import Rule, parse_converter_args

from .config import Config
from .page import PAGES
from .types import ResponseBase, RequestBase, Response
from .utils import parse_multi_dict, parse_rule


# region responses
def make_json_response(
    content: Union[BaseModel, Iterable[BaseModel]],
    status_code: int,
    by_alias: bool,
    exclude_none: bool = False,
    many: bool = False,
) -> FlaskResponse:
    """serializes model, creates JSON response with given status code"""
    if many:
        js = f"[{', '.join([model.model_dump_json(exclude_none=exclude_none, by_alias=by_alias) for model in content])}]"
    else:
        js = content.model_dump_json(exclude_none=exclude_none, by_alias=by_alias)
    response = make_response(js, status_code)
    response.mimetype = "application/json"
    return response


def unsupported_media_type_response(request_cont_type: str) -> FlaskResponse:
    body = {
        "detail": f"Unsupported media type '{request_cont_type}' in request. "
        "'application/json' is required."
    }
    return make_response(jsonify(body), 415)
# endregion

# region utils


def validate_path_params(func: Callable, kwargs: dict, excluded: list[str] = []) -> Tuple[dict, list]:
    errors = []
    validated = {}
    for name, type_ in func.__annotations__.items():
        if name in excluded:
            continue
        try:
            adapter = TypeAdapter(type_)
            validated[name] = adapter.validate_python(kwargs.get(name))
        except ValidationError as e:
            err = e.errors()[0]
            err["loc"] = [name]
            errors.append(err)
    kwargs = {**kwargs, **validated}
    return kwargs, errors


def get_body_dict(**params):
    data = request.get_json(**params)
    if data is None and params.get("silent"):
        return {}
    return data
# endregion


class PydanticValidationErrorWrapper(Exception):
    """This class works as pydantic version 1 validation error class which 
    """

    def __init__(self, model: type[BaseModel], error: ValidationError):
        self.model = model
        self.error = error

    def __getattr__(self, name):
        if hasattr(self.error, name):
            return getattr(self.error, name)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


@dataclass
class Context:
    query: Optional[BaseModel]
    body: Optional[BaseModel]
    headers: Optional[BaseModel]
    cookies: Optional[BaseModel]


class FlaskBackend:
    def __init__(self, validator: Any) -> None:
        self.validator = validator
        self.config: Config = validator.config
        self.logger: logging.Logger = logging.getLogger(__name__)

    def find_routes(self) -> Any:
        for rule in self.app.url_map.iter_rules():
            if any(
                str(rule).startswith(path)
                for path in (f"/{self.config.PATH}", "/static")
            ):
                continue
            yield rule

    def bypass(self, func: Callable, method: str) -> bool:
        if method in ["HEAD", "OPTIONS"]:
            return True
        return False

    def parse_func(self, route: Any) -> Any:
        func = self.app.view_functions[route.endpoint]
        for method in route.methods:
            yield method, func

    def parse_path(self, route: Rule) -> Tuple[str, List[Any]]:

        subs = []
        parameters = []

        for converter, arguments, variable in parse_rule(route):
            if converter is None:
                subs.append(variable)
                continue
            subs.append(f"{{{variable}}}")

            args: Iterable[Any] = []
            kwargs: Dict[str, Any] = {}

            if arguments:
                args, kwargs = parse_converter_args(arguments)

            schema = None
            if converter == "any":
                schema = {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": args,
                    },
                }
            elif converter == "int":
                schema = {
                    "type": "integer",
                    "format": "int32",
                }
                if "max" in kwargs:
                    schema["maximum"] = kwargs["max"]
                if "min" in kwargs:
                    schema["minimum"] = kwargs["min"]
            elif converter == "float":
                schema = {
                    "type": "number",
                    "format": "float",
                }
            elif converter == "uuid":
                schema = {
                    "type": "string",
                    "format": "uuid",
                }
            elif converter == "path":
                schema = {
                    "type": "string",
                    "format": "path",
                }
            elif converter == "string":
                schema = {
                    "type": "string",
                }
                for prop in ["length", "maxLength", "minLength"]:
                    if prop in kwargs:
                        schema[prop] = kwargs[prop]
            elif converter == "default":
                schema = {"type": "string"}

            parameters.append(
                {
                    "name": variable,
                    "in": "path",
                    "required": True,
                    "schema": schema,
                }
            )

        return "".join(subs), parameters

    def request_validation(
        self,
        request: FlaskRequest,
        query: Optional[Type[BaseModel]],
        body: Optional[RequestBase],
        headers: Optional[Type[BaseModel]],
        cookies: Optional[Type[BaseModel]],
    ) -> list[tuple[str, type[BaseModel]]]:

        body_model: type[BaseModel] = getattr(
            body, "model") if body and getattr(body, "model") else None

        def validated_query(model: type[BaseModel] | None):
            if model is None:
                return None

            raw_query = request.args or None
            req_query = parse_multi_dict(raw_query, model=model) if raw_query is not None else {}

            return model.model_validate(req_query)

        def validate_body(model: type[BaseModel] | None):
            if model is None:
                return None

            if request.content_type and "application/json" in request.content_type:
                if request.content_encoding and "gzip" in request.content_encoding:
                    raw_body = gzip.decompress(request.stream.read()).decode(
                        encoding="utf-8"
                    )
                    parsed_body = json.loads(raw_body)
                else:
                    parsed_body = request.get_json() or {}
            elif request.content_type and "multipart/form-data" in request.content_type:
                parsed_body = parse_multi_dict(request.form, model=model) if request.form else {}
            else:
                parsed_body = request.get_data() or {}

            return model.model_validate(parsed_body)

        req_headers: Optional[Headers] = request.headers or None
        req_cookies: Optional[Mapping[str, str]] = request.cookies or None

        items = [
            ('query', query, validated_query),
            ('body', body_model, validate_body),
            ('headers', headers, lambda m: m.model_validate(req_headers or {}) if m else None),
            ('cookies', cookies, lambda m: m.model_validate(req_cookies or {}) if m else None)]

        params: list[tuple[str, type[BaseModel]]] = []
        for key, model, c in items:
            try:
                params.append((key, c(model)))
            except ValidationError as err:
                raise PydanticValidationErrorWrapper(model=model, error=err)

        return params

    def validate(
        self,
        func: Callable,
        query: Optional[Type[BaseModel]],
        body: Optional[RequestBase],
        headers: Optional[Type[BaseModel]],
        cookies: Optional[Type[BaseModel]],
        resp: Optional[ResponseBase],
        before: Callable,
        after: Callable,
        on_success_status: int | None = None,
        response_by_alias: bool = False,
        response_exclude_none: bool = False,
        excluded: list[str] = [],
        *args: List[Any],
        **kwargs: Mapping[str, Any],
    ) -> FlaskResponse:
        """
    Decorator for route methods which will validate query, body, headers, cookies and even decorated method parameters
    as well as serialize the response (if it derives based on `resp` named parameter class).

    Request parameters (query, body, headers, cookies) are accessible directly via `kwargs`, if you define them in the decorated function.

    `on_success_status` status code for success responses. default=200
    `response_by_alias` whether Pydantic's alias is used
    `response_exclude_none` whether to remove None fields from response
    `excluded` - List decorated function parameters that should be exluded when validating parameters

    example::

        from flask import request
        from flask_pydantic import validate
        from pydantic import BaseModel

        class Query(BaseModel):
            query: str

        class Body(BaseModel):
            color: str

        class Form(BaseModel):
            name: str

        class MyModel(BaseModel):
            id: int
            color: str
            description: str

        ...

        @app.route("/")
        @validate(query=Query, body=Body)
        def test_route(query: Query, body: Body):
            query = request.query_params.query
            color = request.body_params.query

            return MyModel(...)

        @app.route("/kwargs")
        @validate(resp=Response(HTTP_200=MyModel))
        def test_route_kwargs():

            return MyModel(...)

    -> that will render JSON response with serialized MyModel instance
    """

        # Check if paths should be validated
        kwargs, err = validate_path_params(func, kwargs, excluded=list(
            set(excluded + ['query', 'body', 'headers', 'cookies'])))
        if err:
            return make_response(
                jsonify(dict(validation_errors=err)), self.config.VALIDATION_ERROR_CODE
            )

        response, req_validation_error, resp_validation_error, params = None, None, None, None
        try:
            params = self.request_validation(request, query, body, headers, cookies)
        except PydanticValidationErrorWrapper as err:
            req_validation_error = err
            response = make_response(
                jsonify(dict(validation_errors=err.error.errors())), self.config.VALIDATION_ERROR_CODE
            )

        before(request, response, req_validation_error, None)
        if req_validation_error:
            abort(response)  # type: ignore

        # for each (body, query, cookies, headers, cookies) validated from the request, add it as parameter
        # to the request
        if params:
            for key, model in params:
                if func.__annotations__.get(key):
                    kwargs[key] = model

        response = make_response(func(*args, **kwargs))

        if isinstance(resp, BaseModel):
            resp = Response(HTTP_200=resp)

        # case response class is provided and response class has model and has model_validate exists on response
        if resp and resp.has_model() and getattr(resp, "validate"):
            model = resp.find_model(response.status_code)
            if model:
                try:
                    model = model.model_validate(response.get_json())
                    response = make_json_response(model, status_code=on_success_status if on_success_status else response.status_code,
                                                  by_alias=response_by_alias, exclude_none=response_exclude_none)
                except ValidationError as err:
                    resp_validation_error = PydanticValidationErrorWrapper(model=model, error=err)
                    response = make_response(jsonify(dict(validation_errors=err.errors())), 500)

        after(request, response, resp_validation_error, None)

        return response

    def register_route(self, app: Flask) -> None:
        self.app = app
        from flask import jsonify

        self.app.add_url_rule(
            self.config.spec_url,
            "openapi",
            lambda: jsonify(self.validator.spec),
        )

        for ui in PAGES:
            self.app.add_url_rule(
                f"/{self.config.PATH}/{ui}",
                f"doc_page_{ui}",
                lambda ui=ui: PAGES[ui].format(self.config),
            )
