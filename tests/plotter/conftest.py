import pathlib
import io
import hashlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from plotter import Plotter

mpl.use("Agg")
HASH_DIR: pathlib.Path = pathlib.Path(__file__).parent / "hashes"
FIGURE_DIR: pathlib.Path = pathlib.Path(__file__).parent / "figures"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Update snapshot hashes and baseline figures under tests/plotter."
    )


def hash_figure(figure: plt.Figure) -> str:
    buf = io.BytesIO()
    figure.savefig(buf, format="png", dpi=100)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def build_paths(test_name: str) -> tuple[pathlib.Path, pathlib.Path]:
    file_name = pathlib.Path(test_name)
    hash_path = HASH_DIR / file_name.with_suffix(".sha256")
    figure_path = FIGURE_DIR / file_name.with_suffix(".png")
    return hash_path, figure_path


@pytest.fixture
def assert_figure(request):
    update = request.config.getoption("--update-snapshots")

    def _assert(figure: plt.Figure) -> None:
        digest = hash_figure(figure)
        hash_path, figure_path = build_paths(request.node.name)

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

    return _assert
