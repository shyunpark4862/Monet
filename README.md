# Monet
Monet is a Python function plotting package built on Matplotlib, designed to produce high-quality, publication-ready scientific graphs with minimal effort. It specializes in intelligent plotting through adaptive sampling and data refinement.

## Key Features
- **Adaptive Refinement**: Automatically increases sampling density in areas of high curvature or around contour lines. This ensures smooth and accurate plots even with a low number of initial samples.
- **Automatic Clipping**: For functions with outliers or asymptotic behavior, Monet can automatically calculate a "focus zone" using the interquartile range (IQR) to adjust the visualization range for clarity.
- **Concise API**: Abstracts away the complexity of Matplotlib settings, providing an intuitive API to generate beautiful plots with minimal code.
- **Scientific Style**: Comes with a pre-configured, high-quality Matplotlib style suitable for scientific papers and presentations.

## Installation
To install Monet, you can clone the repository and install it using pip:

```Bash
git clone https://github.com/shyunpark4862/monet.git
cd monet
pip install
```
For development, install with the `[dev]` extras to include testing libraries like `pytest`:

```Bash
pip install .[dev]
```

## Usage
Here are some basic examples of how to use Monet to create plots.

### Line Plot (`fline`)

Plot a 1D function, like $y = \sin(x) / x$, with adaptive sampling.

```Python
import numpy as np
from fplotter import FPlotter

# Define the function
def sinc_func(x):
    return np.sin(x) / x

# Create a plotter instance and plot the function
plotter = FPlotter()
plotter.fline(sinc_func, xbound=(-10, 10), legend=r"$\frac{\sin(x)}{x}$")

# Customize and show the plot
plotter.title("Adaptive Line Plot")
plotter.axis_label("$x$", "$f(x)$")
plotter.legend()
plotter.show()
```

### Contour Plot (`fcontour`)

Create a contour plot for a 2D function, like $z = \sin(x) * y + \cos(y) * x$.

```Python
import numpy as np
from fplotter import FPlotter

# Define the function
def bivariate_func(x, y):
    return np.sin(x) * y + np.cos(y) * x

# Create and configure the plot
plotter = FPlotter()
plotter.fcontour(bivariate_func, xbound=(-5, 5), ybound=(-5, 5))
plotter.title("Adaptive Contour Plot")
plotter.axis_label("$x$", "$y$")
plotter.show()
```
### Heatmap (`fheatmap`)

Generate a heatmap for the same 2D function, automatically clipping the z-axis values to a viewable range.

```Python
import numpy as np
from fplotter import FPlotter

# Define the function
def bivariate_func(x, y):
    return np.sin(x) * y + np.cos(y) * x

# Create and configure the plot with auto-clipping
plotter = FPlotter()
plotter.fheatmap(
    bivariate_func, 
    xbound=(-5, 5), 
    ybound=(-5, 5),
    auto_clip=True,
    clip_line=True
)
plotter.title("Adaptive Heatmap with Auto-Clipping")
plotter.axis_label("$x$", "$y$")
plotter.show()
```

## Project Structure
Monet is organized into several modules that work together to create the final plots:

- **sampler**: Handles the initial uniform sampling of functions and provides data container classes (Sample, Sample2d, Sample3d).
- **clipper**: Implements the logic for automatically or manually determining data boundaries to clip outliers.
- **refiner**: Contains the core adaptive mesh refinement algorithm, which adds new sample points based on geometric properties like curvature.
- **resampler**: Takes the potentially unstructured data from the refiner and interpolates it back onto a regular grid for plotting.
- **fplotter**: The main user-facing class that orchestrates the sampling, clipping, refining, and plotting process.
- **plotter**: A base class that provides a simplified wrapper around Matplotlib's plotting functions.

## License
This project is licensed under the MIT License. See the LICENSE file for details.