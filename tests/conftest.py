"""Shared pytest setup for the test suite.

Everything in this module runs at import time, before any test module (and
therefore before ``api.models.bedrock``) is imported. That ordering matters:

* ``src`` has to be on ``sys.path`` so ``from api.models import bedrock`` resolves.
* boto3 reads credentials and the region when a client is constructed, and
  ``api.models.bedrock`` constructs its clients at module import time. The dummy
  values below keep that construction offline and away from instance metadata.

No real AWS call is ever expected to succeed in the test suite; tests stub the
boto3 clients they exercise.
"""

import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Overwrite rather than default: whatever the developer happens to have exported
# must not leak into the suite, otherwise results stop being reproducible and a
# test could reach a real account.
os.environ.update(
    {
        "AWS_REGION": "us-west-2",
        "AWS_DEFAULT_REGION": "us-west-2",
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "AWS_EC2_METADATA_DISABLED": "true",
    }
)
