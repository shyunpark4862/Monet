import os
import random

import numpy as np
import pytest


def pytest_configure(config: pytest.Config) -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    random.seed(0)
    np.random.seed(0)
