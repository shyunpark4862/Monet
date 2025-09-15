import hashlib
import io
import os
import random
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from plotter import Plotter

mpl.use("Agg")      # Disable interactive mode


def pytest_configure(config: pytest.Config) -> None:
    """ Sets the random seed. """
    os.environ.setdefault("PYTHONHASHSEED", '0')
    random.seed(0)
    np.random.seed(0)


def pytest_addoption(parser: pytest.Parser) -> None:
    """ Adds the --update-snapshots option. """
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Update snapshot hashes and baseline figures."
    )


def hash_figure(figure: plt.Figure) -> str:
    """ Hashes a figure. """
    buf = io.BytesIO()
    figure.savefig(buf, format="png", dpi=100)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def build_paths(request: pytest.FixtureRequest) -> tuple[Path, Path]:
    """ Builds the paths for the snapshot hash and baseline figure. """
    test_dir = Path(request.node.path).parent
    hash_dir, figure_dir = test_dir / "hashes", test_dir / "figures"
    file_name = Path(request.node.name)
    hash_path = hash_dir / file_name.with_suffix(".sha256")
    figure_path = figure_dir / file_name.with_suffix(".png")
    return hash_path, figure_path


@pytest.fixture
def assert_figure(request):
    """ Assert that a figure matches the snapshot. """
    update = request.config.getoption("--update-snapshots")

    def _assert_figure(figure: plt.Figure) -> None:
        digest = hash_figure(figure)
        hash_path, figure_path = build_paths(request)

        if update:
            hash_path.parent.mkdir(parents=True, exist_ok=True)
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            hash_path.write_text(digest, encoding="utf-8")
            plotter = Plotter()
            plotter.figure = figure
            plotter.save(figure_path)
            pytest.skip("Snapshot updated")
        else:
            if not hash_path.exists():
                pytest.fail(f"Missing snapshot: {hash_path}")
            expected = hash_path.read_text(encoding="utf-8")
            assert digest == expected, (
                f"Snapshot mismatch for {request.node.name}\n"
                f"expected: {expected}\nactual  : {digest}"
            )

    return _assert_figure
