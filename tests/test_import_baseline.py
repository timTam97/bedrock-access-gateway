"""Baseline smoke test for the model discovery module.

``api.models.bedrock`` builds its boto3 clients and calls ``list_bedrock_models()``
at import time. When those calls fail - no credentials, no permission, a brand new
stack - the module is expected to swallow the error and fall back to a single-entry
model list built from ``DEFAULT_MODEL``. This test pins that contract down so later
changes to model discovery cannot quietly remove it.
"""

from api.models import bedrock
from api.setting import DEFAULT_MODEL


class FailingBedrockClient:
    """Stands in for a boto3 ``bedrock`` client that cannot talk to the service."""

    def get_paginator(self, operation_name: str):
        raise RuntimeError(f"stubbed failure for {operation_name}")

    def list_foundation_models(self, **kwargs):
        raise RuntimeError("stubbed failure for list_foundation_models")


def test_list_bedrock_models_falls_back_to_default_model(monkeypatch):
    monkeypatch.setattr(bedrock, "bedrock_client", FailingBedrockClient())

    model_list = bedrock.list_bedrock_models()

    assert model_list == {DEFAULT_MODEL: {"modalities": ["TEXT", "IMAGE"]}}
