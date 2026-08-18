"""Shared test configuration.

Property tests run with a fixed seed in continuous integration and a random one
locally. A fixed seed makes a red build reproducible; a random one keeps the
suite exploring rather than re examining the same hundred cases for ever.
"""

import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci",
    derandomize=True,
    # Continuous integration runners are shared and their timing is not this
    # library's to control. A wall clock deadline on a property test there
    # produces failures that say nothing about the property.
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("dev", deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
