from functools import wraps
from typing import Mapping, Optional, Type, Union, Callable, Iterable, Any, Dict

from flask import Flask, Response as FlaskResponse
from pydantic import BaseModel
from inflection import camelize

from . import Request
from .config import Config
from .flask_backend import FlaskBackend
from .types import RequestBase, ResponseBase
from .utils import (
    parse_comments,
    parse_request,
    parse_params,
    parse_resp,
    parse_name,
    default_before_handler,
    default_after_handler,
    update_open_api_schema_definitions,
)


class FlaskPydanticOpenapi:
    """
    Interface

    :param str backend_name: choose from ('flask')
    :param backend: a backend that inherit `flask_pydantic_openapi.FlaskBackend`
    :param app: backend framework application instance (you can also register to it later)
    :param before: a callback function of the form :meth:`fla.utils.default_before_handler`
        ``func(req, resp, req_validation_error, instance)``
        that will be called after the request validation before the endpoint function
    :param after: a callback function of the form :meth:`spectree.utils.default_after_handler`
        ``func(req, resp, resp_validation_error, instance)``
        that will be called after the response validation
    :param kwargs: update default :class:`spectree.config.Config`
    """

    def __init__(
        self,
        backend_name: str = "base",
        backend: Type[FlaskBackend] = FlaskBackend,
        app: Optional[Flask] = None,
        before: Callable = default_before_handler,
        after: Callable = default_after_handler,
        **kwargs: Any,
    ):
        self.before: Callable = before
        self.after: Callable = after
        self.config = Config(**kwargs)
        self.backend_name = backend_name
        self.backend = backend(self)
        # init
        self.models: Dict[str, Any] = {}
        if app:
            self.register(app)

    def register(self, app: Flask) -> None:
        """
        register to backend application

        This will be automatically triggered if the app is passed into the
        init step.
        """
        self.app = app
        self.backend.register_route(self.app)

    @property
    def spec(self) -> Mapping[str, Any]:
        """
        get the OpenAPI spec
        """
        if not hasattr(self, "_spec"):
            self._spec = self._generate_spec()
        return self._spec

    def bypass(self, func: Callable) -> bool:
        """
        bypass rules for routes (mode defined in config)

        :normal:    collect all the routes that are not decorated by other
                    `SpecTree` instance
        :greedy:    collect all the routes
        :strict:    collect all the routes decorated by this instance
        """
        if self.config.MODE == "greedy":
            return False
        elif self.config.MODE == "strict":
            if getattr(func, "_decorator", None) == self:
                return False
            return True
        else:
            decorator = getattr(func, "_decorator", None)
            if decorator and decorator != self:
                return True
            return False

    def validate(
        self,
        query: Optional[Type[BaseModel]] = None,
        body: Optional[Union[RequestBase, Type[BaseModel]]] = None,
        headers: Optional[Type[BaseModel]] = None,
        cookies: Optional[Type[BaseModel]] = None,
        resp: Optional[ResponseBase] = None,
        tags: Iterable[str] = (),
        deprecated: bool = False,
        before: Optional[Callable] = None,
        after: Optional[Callable] = None,
        on_success_status: int | None = None,
        response_by_alias: bool = False,
        response_exclude_none: bool = False,
        excluded: list[str] = [],
    ) -> Callable:
        """
        - validate query, body, headers in request
        - validate response body and status code
        - add tags to this API route

        :param query: `pydantic.BaseModel`, query in uri like `?name=value`
        :param body: `spectree.Request`, Request body
        :param headers: `pydantic.BaseModel`, if you have specific headers
        :param cookies: `pydantic.BaseModel`, if you have cookies for this route
        :param resp: `spectree.Response`
        :param tags: a tuple of tags string
        :param deprecated: You can mark specific operations as deprecated to indicate
         that they should be transitioned out of usage
        :param before: :meth:`spectree.utils.default_before_handler` for specific endpoint
        :param after: :meth:`spectree.utils.default_after_handler` for specific endpoint
        :param on_success_status: status code for success responses. default=200
        :param response_by_alias: whether Pydantic's alias is used
        :param response_exclude_none: whether to remove None fields from response
        :param excluded: List decorated function parameters that should be exluded when validating parameters
        """

        def decorate_validation(func: Callable) -> Callable:
            @wraps(func)
            def sync_validate(*args: Any, **kwargs: Any) -> FlaskResponse:
                return self.backend.validate(
                    func,
                    query,
                    body if isinstance(body, RequestBase) else Request(body),
                    headers,
                    cookies,
                    resp,
                    before or self.before,
                    after or self.after,
                    on_success_status,
                    response_by_alias,
                    response_exclude_none,
                    excluded,
                    *args,
                    **kwargs,
                )

            validation = sync_validate

            # register
            for name, model in zip(
                ("query", "body", "headers", "cookies"), (query, body, headers, cookies)
            ):
                if model is not None:
                    if hasattr(model, "model"):
                        _model = getattr(model, "model", None)
                    else:
                        _model = model
                    if _model:
                        self.models = update_open_api_schema_definitions(
                            self.models,
                            _model.model_json_schema(ref_template='#/components/schemas/{model}'),
                            _model.__name__
                        )
                    setattr(validation, name, model)

            if resp:
                for model in resp.models:
                    if model:
                        assert not isinstance(model, RequestBase)
                        self.models = update_open_api_schema_definitions(
                            self.models,
                            model.model_json_schema(ref_template='#/components/schemas/{model}'),
                            model.__name__
                        )
                setattr(validation, "resp", resp)

            if tags:
                setattr(validation, "tags", tags)

            if deprecated:
                setattr(validation, "deprecated", True)

            # register decorator
            setattr(validation, "_decorator", self)
            return validation

        return decorate_validation

    def _generate_spec(self) -> Mapping[str, Any]:
        """
        generate OpenAPI spec according to routes and decorators
        """
        if not self.config.VISIBLE:
            return {}

        tag_lookup = {tag["name"]: tag for tag in self.config.TAGS}
        routes: Dict[str, Any] = {}
        tags: Dict[str, Any] = {}
        for route in self.backend.find_routes():
            path, parameters = self.backend.parse_path(route)
            routes[path] = routes.get(path, {})
            for method, func in self.backend.parse_func(route):
                if self.backend.bypass(func, method) or self.bypass(func):
                    continue

                name = parse_name(func)
                summary, desc = parse_comments(func)
                func_tags = getattr(func, "tags", ())
                for tag in func_tags:
                    if tag not in tags:
                        tags[tag] = tag_lookup.get(tag, {"name": tag})

                routes[path][method.lower()] = {
                    "summary": summary or f"{name} <{method}>",
                    "operationId": camelize(f"{name}", False),
                    "description": desc or "",
                    "tags": getattr(func, "tags", []),
                    "parameters": parse_params(func, parameters[:], self.models),
                    "responses": parse_resp(func, self.config.VALIDATION_ERROR_CODE),
                }
                if hasattr(func, "deprecated"):
                    routes[path][method.lower()]["deprecated"] = True

                request_body = parse_request(func)
                if request_body:
                    routes[path][method.lower()][
                        "requestBody"
                    ] = self._parse_request_body(request_body)

        for route in list(routes.keys()):
            routes[f"{self.config.ROOT_PATH}{route}"] = routes.pop(route)

        spec = {
            "openapi": self.config.OPENAPI_VERSION,
            "info": {
                **self.config.INFO,
                **{
                    "title": self.config.TITLE,
                    "version": self.config.VERSION,
                },
            },
            "tags": list(tags.values()),
            "paths": {**routes},
            "components": {"schemas": {**self._get_model_definitions()}},
        }
        return spec

    def _get_model_definitions(self) -> Dict[str, Any]:
        """
        handle nested models
        """
        definitions: Dict[str, Any] = {}
        for model, schema in self.models.items():
            if model not in definitions.keys():
                definitions[model] = schema
            if "definitions" in schema:
                for key, value in schema["definitions"].items():
                    definitions = update_open_api_schema_definitions(definitions, value, key)
                del schema["definitions"]

        return definitions

    def _parse_request_body(self, request_body: Mapping[str, Any]) -> Mapping[str, Any]:
        content_types = list(request_body["content"].keys())
        if len(content_types) != 1:
            raise RuntimeError(
                "Cannot currently handle multiple content types for a single request"
            )
        else:
            content_type = content_types[0]
        schema = request_body["content"][content_type]["schema"]
        if "$ref" not in schema.keys():
            # handle inline schema definitions
            schema_ = dict()
            schema_ = update_open_api_schema_definitions(schema_, schema, 'schema')
            return {
                "content": {content_type: schema_}
            }
        else:
            return request_body
