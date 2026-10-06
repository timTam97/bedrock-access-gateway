"""Tests for cross-region inference profile discovery in ``list_bedrock_models()``.

The function combines three sources: on-demand foundation models, system-defined
(cross-region) inference profiles and application inference profiles. These tests
stub the boto3 ``bedrock`` client so the combination logic can be exercised without
credentials, and assert on the dict the function actually returns.

Every base model ID used below is a real Converse-reachable Anthropic model ID. The
``responseStreamingSupported`` / ``modelLifecycle`` flags and the exact set of
profile IDs returned per model are fixture values chosen to exercise the code paths,
not a claim about what any particular Region offers today.
"""

from api.models import bedrock

OPUS_4_6 = "anthropic.claude-opus-4-6-v1"
OPUS_4_5 = "anthropic.claude-opus-4-5-20251101-v1:0"
SONNET_4_6 = "anthropic.claude-sonnet-4-6"
SONNET_4_5 = "anthropic.claude-sonnet-4-5-20250929-v1:0"
HAIKU_4_5 = "anthropic.claude-haiku-4-5-20251001-v1:0"
OPUS_4_1 = "anthropic.claude-opus-4-1-20250805-v1:0"
HAIKU_3_5 = "anthropic.claude-3-5-haiku-20241022-v1:0"

SYSTEM_DEFINED_PROFILE_IDS = [
    # Opus 4.6 under five prefixes. "jp." and "au." are exactly the prefixes the old
    # single-guessed-prefix code could never produce. Both are real prefixes
    # documented by AWS; "au." is included here purely to prove the discovery is
    # prefix-agnostic, since the production code only ever lists what
    # ListInferenceProfiles returned.
    "us." + OPUS_4_6,
    "eu." + OPUS_4_6,
    "jp." + OPUS_4_6,
    "au." + OPUS_4_6,
    "global." + OPUS_4_6,
    # Haiku 4.5 - exercises "global." alongside an on-demand model.
    "us." + HAIKU_4_5,
    "global." + HAIKU_4_5,
    # Sonnet 4.6 deliberately gets only two of its prefixes, so the test can assert
    # the missing ones are not invented.
    "us." + SONNET_4_6,
    "global." + SONNET_4_6,
    # A profile for a model that is not in the foundation model list at all.
    "us." + OPUS_4_5,
    # Sonnet 4.5 is deprecated and reported with a LEGACY lifecycle status here.
    "global." + SONNET_4_5,
    # Profiles for models excluded by the streaming / lifecycle filters.
    "us." + OPUS_4_1,
    "us." + HAIKU_3_5,
    # A profile ID whose prefix is not a single token. Splitting once means this
    # never maps onto SONNET_4_6, where a suffix match would have accepted it.
    "us.unexpected." + SONNET_4_6,
]

FOUNDATION_MODELS = [
    {
        # Profile-only: no ON_DEMAND entitlement, as with the newer Claude models.
        "modelId": OPUS_4_6,
        "inferenceTypesSupported": ["INFERENCE_PROFILE"],
        "inputModalities": ["TEXT", "IMAGE"],
        "responseStreamingSupported": True,
        "modelLifecycle": {"status": "ACTIVE"},
    },
    {
        "modelId": HAIKU_4_5,
        "inferenceTypesSupported": ["ON_DEMAND", "INFERENCE_PROFILE"],
        "inputModalities": ["TEXT", "IMAGE"],
        "responseStreamingSupported": True,
        "modelLifecycle": {"status": "ACTIVE"},
    },
    {
        "modelId": SONNET_4_6,
        "inferenceTypesSupported": ["ON_DEMAND"],
        "inputModalities": ["TEXT", "IMAGE"],
        "responseStreamingSupported": True,
        "modelLifecycle": {"status": "ACTIVE"},
    },
    {
        # Excluded entirely: not stream capable.
        "modelId": OPUS_4_1,
        "inferenceTypesSupported": ["ON_DEMAND"],
        "inputModalities": ["TEXT"],
        "responseStreamingSupported": False,
        "modelLifecycle": {"status": "ACTIVE"},
    },
    {
        # Kept: LEGACY is still a serviceable lifecycle status.
        "modelId": SONNET_4_5,
        "inferenceTypesSupported": ["ON_DEMAND"],
        "inputModalities": ["TEXT", "IMAGE"],
        "responseStreamingSupported": True,
        "modelLifecycle": {"status": "LEGACY"},
    },
    {
        # Excluded entirely: lifecycle status outside ACTIVE / LEGACY.
        "modelId": HAIKU_3_5,
        "inferenceTypesSupported": ["ON_DEMAND"],
        "inputModalities": ["TEXT"],
        "responseStreamingSupported": True,
        "modelLifecycle": {"status": "EOL"},
    },
]

APPLICATION_PROFILE_ARN = "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abcdef123456"

APPLICATION_PROFILES = [
    {
        "inferenceProfileArn": APPLICATION_PROFILE_ARN,
        "models": [
            {
                "modelArn": f"arn:aws:bedrock:us-west-2::foundation-model/{HAIKU_4_5}",
            }
        ],
    }
]


class FakePaginator:
    """Mimics the ``list_inference_profiles`` paginator, one summary per page."""

    def __init__(self, summaries_by_type: dict):
        self.summaries_by_type = summaries_by_type

    def paginate(self, maxResults=None, typeEquals=None):
        for summary in self.summaries_by_type.get(typeEquals, []):
            yield {"inferenceProfileSummaries": [summary]}


class FakeBedrockClient:
    """Stands in for a boto3 ``bedrock`` client with a fixed inventory."""

    def __init__(self, profile_ids=None, foundation_models=None, app_profiles=None):
        self.profile_ids = SYSTEM_DEFINED_PROFILE_IDS if profile_ids is None else profile_ids
        self.foundation_models = FOUNDATION_MODELS if foundation_models is None else foundation_models
        self.app_profiles = [] if app_profiles is None else app_profiles

    def get_paginator(self, operation_name: str):
        assert operation_name == "list_inference_profiles"
        return FakePaginator(
            {
                "SYSTEM_DEFINED": [{"inferenceProfileId": profile_id} for profile_id in self.profile_ids],
                "APPLICATION": self.app_profiles,
            }
        )

    def list_foundation_models(self, **kwargs):
        assert kwargs == {"byOutputModality": "TEXT"}
        return {"modelSummaries": self.foundation_models}


def discover(monkeypatch, **kwargs) -> dict:
    monkeypatch.setattr(bedrock, "bedrock_client", FakeBedrockClient(**kwargs))
    return bedrock.list_bedrock_models()


def test_regional_profile_prefixes_are_taken_from_the_api_response(monkeypatch):
    """Every prefix the API returned is listed, including jp. and au."""
    model_list = discover(monkeypatch)

    for prefix in ("us", "eu", "jp", "au"):
        profile_id = f"{prefix}.{OPUS_4_6}"
        assert profile_id in model_list, profile_id
        assert model_list[profile_id] == {"modalities": ["TEXT", "IMAGE"]}


def test_global_profile_is_still_listed(monkeypatch):
    """Regression guard for the global cross-region support added in 66cb51b."""
    model_list = discover(monkeypatch)

    assert "global." + OPUS_4_6 in model_list
    assert "global." + HAIKU_4_5 in model_list
    assert "global." + SONNET_4_6 in model_list


def test_profile_prefixes_are_never_invented(monkeypatch):
    """A prefix absent from the API response must not be synthesised."""
    model_list = discover(monkeypatch)

    # Sonnet 4.6 was returned under us. and global. only.
    assert "us." + SONNET_4_6 in model_list
    assert "eu." + SONNET_4_6 not in model_list
    assert "jp." + SONNET_4_6 not in model_list
    # Nothing in the fixture offers apac. for any model.
    assert not any(model_id.startswith("apac.") for model_id in model_list)
    # A multi-token prefix is not a cross-region profile for this model.
    assert "us.unexpected." + SONNET_4_6 not in model_list


def test_profiles_do_not_leak_between_models(monkeypatch):
    """A profile only ever attaches to the model its ID actually names."""
    model_list = discover(monkeypatch)

    # Opus 4.5 has a profile but no foundation model entry, so it is not listed.
    assert "us." + OPUS_4_5 not in model_list
    assert OPUS_4_5 not in model_list
    # Opus 4.6's profiles did not pick up Haiku 4.5's or Sonnet 4.6's.
    opus_profiles = {model_id for model_id in model_list if model_id.endswith(OPUS_4_6)}
    assert opus_profiles == {
        "us." + OPUS_4_6,
        "eu." + OPUS_4_6,
        "jp." + OPUS_4_6,
        "au." + OPUS_4_6,
        "global." + OPUS_4_6,
    }


def test_profile_only_model_is_listed_without_its_bare_id(monkeypatch):
    model_list = discover(monkeypatch)

    # No ON_DEMAND entitlement, so the bare ID is not callable...
    assert OPUS_4_6 not in model_list
    # ...but the profiles are.
    assert "us." + OPUS_4_6 in model_list


def test_on_demand_model_is_listed_under_its_bare_id(monkeypatch):
    model_list = discover(monkeypatch)

    assert model_list[HAIKU_4_5] == {"modalities": ["TEXT", "IMAGE"]}


def test_legacy_lifecycle_models_are_still_listed(monkeypatch):
    """LEGACY is serviceable, unlike EOL - a deprecated model must stay callable."""
    model_list = discover(monkeypatch)

    assert SONNET_4_5 in model_list
    assert "global." + SONNET_4_5 in model_list


def test_non_streaming_and_eol_models_are_excluded_entirely(monkeypatch):
    model_list = discover(monkeypatch)

    for model_id in (OPUS_4_1, HAIKU_3_5):
        assert model_id not in model_list
        assert "us." + model_id not in model_list


def test_cross_region_inference_can_be_disabled(monkeypatch):
    # The flag is imported by value at module load, so patch the module attribute.
    monkeypatch.setattr(bedrock, "ENABLE_CROSS_REGION_INFERENCE", False)

    model_list = discover(monkeypatch)

    assert not any(model_id.startswith(("us.", "eu.", "jp.", "au.", "global.")) for model_id in model_list)
    # On-demand models are unaffected.
    assert HAIKU_4_5 in model_list
    assert SONNET_4_6 in model_list


def test_application_inference_profile_arns_are_still_listed(monkeypatch):
    model_list = discover(monkeypatch, app_profiles=APPLICATION_PROFILES)

    assert model_list[APPLICATION_PROFILE_ARN] == {"modalities": ["TEXT", "IMAGE"]}


def test_application_inference_profiles_can_be_disabled(monkeypatch):
    monkeypatch.setattr(bedrock, "ENABLE_APPLICATION_INFERENCE_PROFILES", False)

    model_list = discover(monkeypatch, app_profiles=APPLICATION_PROFILES)

    assert APPLICATION_PROFILE_ARN not in model_list
