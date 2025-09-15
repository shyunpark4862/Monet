from typing import Literal
from collections.abc import Iterable

import numpy as np
from pydantic import AfterValidator
from typing_extensions import Annotated

Color = str | tuple[float, float, float]
LineStyle = Literal["solid", "dashed", "dashdot", "dotted"]