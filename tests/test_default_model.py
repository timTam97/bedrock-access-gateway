"""Keeps the three places that carry a default model ID in step.

``DEFAULT_MODEL`` in src/api/setting.py and the ``DefaultModelId`` parameter of both
CloudFormation templates are edited by hand, so they drift silently. A deployment
that leaves the parameter alone must get the same model as a container run with no
``DEFAULT_MODEL`` set.
"""

from pathlib import Path

import pytest

from api.setting import DEFAULT_MODEL

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = [
    REPO_ROOT / "deployment" / "BedrockProxy.template",
    REPO_ROOT / "deployment" / "BedrockProxyFargate.template",
]


def default_model_id(template: Path) -> str:
    """Read the Default of the DefaultModelId parameter out of a CFN template."""
    lines = template.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "DefaultModelId:":
            for following in lines[index + 1 :]:
                if not following.startswith("    "):
                    break
                key, _, value = following.strip().partition(":")
                if key == "Default":
                    return value.strip()
    raise AssertionError(f"No DefaultModelId default found in {template.name}")


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda path: path.name)
def test_template_default_matches_setting_default(template):
    assert default_model_id(template) == DEFAULT_MODEL
