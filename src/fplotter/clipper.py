import numpy as np

from . import sampler


def clip(
        sample: sampler.Sample,
        bound: tuple[float, float] | None,
        k: float
) -> tuple[np.ndarray, tuple[float, float] | None]:
    """
    Creates a mask to identify data points that fall outside a given boundary.

    This function identifies values in the last column of the data (the 
    function's output, y) that fall outside a specified upper and lower bound. 
    If a boundary is not explicitly provided, it is calculated automatically 
    using a statistical approach based on the interquartile range (IQR).
    
    Parameters
    ----------
    sample : sampler.Sample
        A ``sampler.Sample`` object containing the data points. The mask is
        generated based on the data in the last column. The sample must not be
        empty.
    bound : tuple[float, float] or None
        The (lower, upper) boundary for filtering the data. If this value is
        None, the boundary is calculated automatically. When provided, the first
        element must be smaller than the second element.
    k : float
        The IQR coefficient used when automatically calculating boundaries if
        ``bound`` is None.

    Returns
    -------
    np.ndarray
        A boolean mask array. Values outside the boundary are marked as True,
        and values inside are False.
    tuple[float, float] or None
        The (lower, upper) boundary used for clipping. None if no valid
        boundary could be computed.

    Notes
    -----
    This function does not modify the input ``sample``, it only returns a mask 
    array.
    """
    y = sample.data[:, -1]
    if bound is None:
        bound = _compute_focus_zone(y, k)
    if bound is None:
        return np.repeat(False, len(y)), bound
    return (y < bound[0]) | (y > bound[1]), bound


def _compute_focus_zone(
        y: np.ndarray,
        k: float
) -> tuple[float, float] | None:
    """
    Calculates the focus zone of data using the interquartile range (IQR).

    This function uses the Tukey's fences method to determine the statistical
    central range of the data. The calculated range is
    [Q1 - k * IQR, Q3 + k * IQR], where Q1 is the first quartile, Q3 is the
    third quartile, and IQR is the interquartile range (Q3 - Q1). For more
    details, see:

    - Method overview: https://en.wikipedia.org/wiki/Outlier
    - Original paper: Tukey, John Wilder. Exploratory data analysis. Vol. 2.
                      Reading, MA: Addison-wesley, 1977.

    Parameters
    ----------
    y : np.ndarray of shape (n_samples,)
        The 1D data array on which to perform the calculation. NaN values in
        the array are automatically ignored.
    k : float
        The coefficient to be multiplied by the IQR. This value controls the
        sensitivity of outlier detection.

    Returns
    -------
    tuple[float, float] or None
        A tuple containing the (lower, upper) of the calculated focus zone.
        Returns None if all data is NaN or the IQR is close to zero, making it
        impossible to calculate a valid range.

    """
    if np.isnan(y).all():
        return None
    q1, q3 = np.nanpercentile(y, (25, 75))
    iqr = q3 - q1
    if np.isclose(k * iqr, 0, atol=1e-6):
        return None
    return q1 - k * iqr, q3 + k * iqr
