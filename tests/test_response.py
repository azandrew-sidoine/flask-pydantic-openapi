import pytest

from flask_pydantic_openapi.types import (
    DEFAULT_CODE_DESC,
    Response,
    FileResponse,
    Request,
    MultipartFormRequest,
)

from .common import DemoModel


class NormalClass:
    pass


def test_init_response():
    for args, kwargs in [
        ([200], {}),
        (["HTTP_110"], {}),
        ([], {"HTTP_200": NormalClass}),
    ]:
        with pytest.raises(AssertionError):
            Response(*args, **kwargs)

    resp = Response("HTTP_200", HTTP_201=DemoModel)
    assert resp.has_model()
    assert resp.find_model(201) == DemoModel
    assert DemoModel in resp.models

    resp = Response(HTTP_200=None, HTTP_403=DemoModel)
    assert resp.has_model()
    assert resp.find_model(403) == DemoModel
    assert resp.find_model(200) is None
    assert DemoModel in resp.models

    assert not Response().has_model()


def test_response_spec():
    resp = Response("HTTP_200", HTTP_201=DemoModel)
    spec = resp.generate_spec()
    assert spec["200"]["description"] == DEFAULT_CODE_DESC["HTTP_200"]
    assert spec["201"]["description"] == DEFAULT_CODE_DESC["HTTP_201"]
    assert (
        spec["201"]["content"]["application/json"]["schema"]["$ref"].split("/")[-1]
        == "DemoModel"  # noqa: W503
    )

    assert spec.get(200) is None
    assert spec.get(404) is None


def test_file_response_spec():
    octet_resp = FileResponse()
    spec = octet_resp.generate_spec()
    assert spec["200"]["description"] == DEFAULT_CODE_DESC["HTTP_200"]
    assert spec["404"]["description"] == DEFAULT_CODE_DESC["HTTP_404"]

    assert (
        spec["200"]["content"]["application/octet-stream"]["schema"]["format"]
        == "binary"  # noqa: W503
    )
    assert (
        spec["200"]["content"]["application/octet-stream"]["schema"]["type"] == "string"
    )

    pdf_resp = FileResponse("application/pdf")
    pdf_spec = pdf_resp.generate_spec()
    assert pdf_spec["200"]["description"] == DEFAULT_CODE_DESC["HTTP_200"]
    assert pdf_spec["404"]["description"] == DEFAULT_CODE_DESC["HTTP_404"]

    assert pdf_spec["200"]["content"]["application/pdf"]["schema"]["format"] == "binary"
    assert pdf_spec["200"]["content"]["application/pdf"]["schema"]["type"] == "string"


def test_file_request_spec():
    file_request = Request(content_type="application/octet-stream")
    spec = file_request.generate_spec()
    assert spec["content"] == {
        "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
    }


def test_multipart_form_spec():
    form = MultipartFormRequest(DemoModel, "fileName")
    spec = form.generate_spec()
    assert spec["content"] == {
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "properties": {
                    "uid": {
                        "type": "integer",
                        "title": "Uid",
                    },
                    "limit": {"type": "integer", "title": "Limit"},
                    "name": {"type": "string", "title": "Name"},
                    "fileName": {"type": "string", "format": "binary"},
                },
            }
        }
    }


def test_multipart_form_no_model():
    form = MultipartFormRequest()
    spec = form.generate_spec()
    assert spec["content"] == {
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "format": "binary"},
                },
            }
        }
    }
