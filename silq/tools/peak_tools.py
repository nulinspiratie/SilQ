import numpy as np
import scipy
import peakutils
from typing import List, Sequence, Tuple, Union
from matplotlib.axes._axes import Axes

from silq.tools import property_ignore_setter
from qcodes import MatPlot, DataArray
from qcodes.instrument.parameter import MultiParameter


def get_peaks(
    DC_scan: np.ndarray,
    x_vals: Sequence[float] = None,
    threshold: float = 0.5,
    min_dist: int = 10,
) -> List[List[int]]:
    """Extract all peaks in a DC scan.

    Args:
        DC_scan: 2D array containing peaks
        x_vals: set values of the second array dimension.
            If not None, peaks are given based on sweep values
            If None, the array indices will be used for peaks.
        thres: Normalized threshold.
            Only peaks with amplitude higher than the threshold will be detected.
        min_dist: Minimum distance between each detected peak.
            The peak with the highest amplitude is preferred to satisfy this constraint.

    Returns:
        A list where each element is an array of peak indices.
        Note that each element may have a different length.
    """
    peaks = [
        peakutils.indexes(DC_row, thres=threshold, min_dist=min_dist)
        for DC_row in DC_scan
    ]

    if x_vals is not None:
        peaks = [x_vals[peak_arr] for peak_arr in peaks]

    return peaks


def group_peaks(
    peaks_list: List[np.ndarray],
    x_vals: Sequence[float] = None,
    y_vals: Sequence[float] = None,
    slope_estimate: float = None,
    max_distance: Tuple[Union[int, float]] = (4, 4),
    min_consecutive_peaks: int = 4,
) -> List[List[Tuple[float, float]]]:
    """Group peaks of successive lines based on their distance
    The algorithm starts at the first row, and groups peaks of successive rows
    together if their x, y distance is below a certain threshold

    Args:
        peaks_list: list of peak arrays.
            Can be extracted from a 2D map using `get_peaks`.
        x_vals: Optional list of x sweep values (columns).
            Only necessary if max_distance of x is an integer N, in which case
            it will be converted to N*step, where step is the difference of
            successive x_vals.
        y_vals: Optional list of y sweep values (rows).
            If not provided, will use row indices as y_vals
        slope_estimate: Optional estimate of peak slope dy/dx
            Used as a bias for successive rows when computing the max_distance.
            If not provided, a vertical slope is assumed.
        max_distance: maximum distance between peaks
            The first index is the maximum distance along the second dimension (x)
            The second index is the maximum (y)
            If an int is provided, this corresponds to array indices
        min_consecutive_peaks: Minimum number of consecutive peaks

    Returns:

    """
    if y_vals is None:  # y_val corresponds to row index if not provided
        y_vals = np.arange(len(peaks_list))

    # Scale max_distance by x_vals/y_vals step if an int is provided
    max_distance = list(max_distance)
    for k, (max_dist, vals) in enumerate(zip(max_distance, (x_vals, y_vals))):
        if isinstance(max_dist, int):
            if vals is None:
                raise SyntaxError("Must provide x_vals/yvals if max_distance is an int")
            max_distance[k] = abs(max_dist * (vals[1] - vals[0]))

    # Iterate over all peaks in a row, try to group them together
    peak_groups = []
    for k, (y_peak, peaks) in enumerate(zip(y_vals, peaks_list)):  # Loop over rows
        for x_peak in peaks:  # Loop over peaks in a row
            for peak_group in peak_groups:  # Loop over existing peak groups
                # Select last peak of group
                last_group_peak = peak_group[-1]

                # Calculate distance between peak and last peak of group
                delta_x = x_peak - last_group_peak[0]
                delta_y = y_peak - last_group_peak[1]

                # Subtract estimated slope from delta_x if provided
                if slope_estimate is not None:
                    delta_x -= delta_y / slope_estimate

                if abs(delta_x) > max_distance[0] or delta_y > max_distance[1]:
                    # This peak is too far from the peak group
                    continue
                else:
                    # This peak belongs to the peak group
                    peak_group.append((x_peak, y_peak))
                    break
            else:  # No peak groups are close enough, create new peak group
                peak_groups.append([(x_peak, y_peak)])

    # Filter out peak groups with fewer peaks than min_consecutive_peaks
    peak_groups = [
        np.array(peak_group).transpose()
        for peak_group in peak_groups
        if len(peak_group) >= min_consecutive_peaks
    ]

    return peak_groups


def linear_regression(coordinates: Sequence[Tuple[float, float]]) -> dict:
    """Determine optimal line of coordinates using linear regression

    Args:
        coordinates: List of (x,y) coordinates

    Returns:
        Dictionary containing:
            "slope": Optimal slope dy/dx
            "intercept": Optimal intercept
            "start_coords": (x, y) coordinates of line start,
                where y is the minimum of the y coordinates
            "stop_coords": (x, y) coordinates of line stop,
                where y is the maximum of the y coordinates
        Values are NaN if slope is perfectly vertical
    """

    # Apply linear regression to find intercept and slope
    line_regress_result = scipy.stats.linregress(*coordinates)

    result = {
        "slope": line_regress_result.slope,
        "intercept": line_regress_result.intercept,
    }

    # Find start coordinates
    y_start = coordinates[1][0]
    x_start = (y_start - result["intercept"]) / result["slope"]
    result["start_coords"] = (x_start, y_start)

    # Find stop coordinates
    y_stop = coordinates[1][-1]
    x_stop = (y_stop - result["intercept"]) / result["slope"]
    result["stop_coords"] = (x_stop, y_stop)

    return result


def find_nearest_line(
    coords: Tuple[float, float],
    lines: List[dict],
    slope_range: Tuple[float, float] = None,
    y_max_start=None,
    x_distance_range=None,
) -> Union[dict, None]:
    """

    Args:
        coords: (x, y) coord to look for nearest line
        lines: List of lines, each obtained from function `linear_regression`
        slope_range: Optional minimum / maximum slope
        y_max_start: Optional filter out lines starting after this y value

    Returns:
        Line that is closest to x (coords[0]) at y (coords[1]).
        Returns None if no lines are within slope_range or start below y_max_start,

    """
    # filter lines with decent slope
    if slope_range is not None:
        lines = [
            line for line in lines if slope_range[0] <= line["slope"] <= slope_range[1]
        ]

    # filter lines that start above y_max_start
    if y_max_start is not None:
        lines = [line for line in lines if line["start_coords"][1] <= y_max_start]

    # filter lines with tuning points away from edges
    # if min_tuning_edge_distance is not None:
    #     window = np.array(window)  # To allow subtraction by tuple
    #     lines = [
    #         line
    #         for line in lines
    #         if min_tuning_edge_distance
    #         <= min(*line["stop_coords"], *(window - line["stop_coords"]))
    #     ]

    if not lines:
        return None

    x_distances = []
    for line in lines:
        # Find x distance between line at y coord and x coord
        # y = a*x + b -> x = (y - b) / a
        x_line = (coords[1] - line["intercept"]) / line["slope"]
        x_distance = coords[0] - x_line
        x_distances.append(x_distance)
        line["x_distance"] = x_distance

    if (
        x_distance_range is None
        or x_distance_range[0] <= np.min(np.abs(x_distances)) <= x_distance_range[1]
    ):
        nearest_line_index = int(np.argmin(np.abs(x_distances)))
        nearest_line = lines[nearest_line_index]

        return nearest_line
    else:
        return None


def analyze_2D_DC_scan(
    DC_scan: np.ndarray,
    x_vals: Sequence[float] = None,
    y_vals: Sequence[float] = None,
    coords: Tuple[float, float] = None,
    get_peaks_settings: dict = {},
    group_peaks_settings: dict = {},
    find_nearest_line_settings: dict = {},
    plot: Union[bool, Axes] = False,
):
    """Extract Coulomb peaks and their slopes from a 2D DC scan

    Args:
        DC_scan: 2D DC scan exhibiting Coulomb peaks
        x_vals: Sweep values along a row
        y_vals: Sweep values along a column
        coords: Coordinates, used for finding the nearest Coulomb peak
        get_peaks_settings: Settings for the function `get_peaks`. Can be::
            ``threshold`` and ``min_dist``
        group_peaks_settings: Settings for the function `group_peaks`. Can be:
            ``slope_estimate``, ``max_distance``, ``min_consecutive_peaks``:
        find_nearest_line_settings: Settings for function `find_nearest_line`.
            Can be: ``slope_range``, `y_max_start``
        plot: Whether to plot the analyzed DC scan, including the lines for all
            the Coulomb peaks.
            Can either be set to True, or provide a matplotlib Axis

    Returns:

    """
    # Obtain x_vals and y_vals from DC_scan if it is a QCoDeS DataArray
    if x_vals is None and y_vals is None and isinstance(DC_scan, DataArray):
        x_vals = DC_scan.set_arrays[1][0]
        y_vals = DC_scan.set_arrays[0]

    # Get a list where each element is an array with peaks for a given DC scan row
    peaks_list = get_peaks(DC_scan=DC_scan, x_vals=x_vals, **get_peaks_settings)

    # Group all the peaks based on their distance in successive rows
    peak_groups = group_peaks(
        peaks_list=peaks_list, x_vals=x_vals, y_vals=y_vals, **group_peaks_settings
    )

    # Perform linear fit for each group of peaks
    peak_lines = [linear_regression(coordinates) for coordinates in peak_groups]
    # Remove all lines that are perfectly horizontal or vertical
    peak_lines = [line for line in peak_lines if not np.isnan(line["slope"])]

    # # Find pairs of lines that match a charge transition
    # transition_lines = extract_transitions_from_peak_lines(peak_lines, x_vals=x_vals, y_vals=y_vals)

    # Choose center of DC scan as coords if not provided
    if coords is None:
        coords = (np.mean(x_vals), np.mean(y_vals))

    # Find the line that is nearest to the coords
    nearest_line = find_nearest_line(
        coords=coords, lines=peak_lines, **find_nearest_line_settings
    )

    slope = nearest_line["slope"] if nearest_line is not None else None

    results = {
        "peaks_list": peaks_list,
        "peak_groups": peak_groups,
        "peak_lines": peak_lines,
        "coords": coords,
        "nearest_line": nearest_line,
        "slope": slope,
    }

    if plot:
        if isinstance(plot, Axes):
            ax = plot
        else:
            plot_object = MatPlot()
            ax = plot_object[0]
            results["plot_object"] = plot_object

        if x_vals is not None and y_vals is not None:
            ax.add(DC_scan, x=x_vals, y=y_vals)
        else:
            ax.add(DC_scan)

        for peak_group, line in zip(peak_groups, peak_lines):
            ax.plot(
                *zip(line["start_coords"], line["stop_coords"]),
                "-",
                color="g" if line == nearest_line else "w"
            )
            ax.plot(*peak_group, "o", ms=1)

        ax.plot(*coords, "*w")

    return results
