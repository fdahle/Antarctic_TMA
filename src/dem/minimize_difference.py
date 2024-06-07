import numpy as np
from scipy.optimize import minimize


def minimize_difference(reference_dem, other_dem):
    """
    Adjust the SfM DEM by a constant value to minimize the difference with the reference DEM.

    Parameters:
    - reference_dem: 2D numpy array representing the reference DEM.
    - other_dem: 2D numpy array representing the DEM to minimize.

    Returns:
    - adjusted_sfm_dem: 2D numpy array of the adjusted SfM DEM.
    - adjustment_value: The constant value added to the SfM DEM.
    """

    # replace nan values with 0
    reference_dem = np.nan_to_num(reference_dem)
    other_dem = np.nan_to_num(other_dem)

    def mse_difference(c):

        """Calculate the mean squared error between the adjusted SfM DEM and the reference DEM."""
        _adjusted_dem = other_dem + c

        # Create a mask to identify valid (non-NaN) values in both DEMs
        mask = ~np.isnan(reference_dem) & ~np.isnan(_adjusted_dem)

        mse = np.mean((reference_dem[mask] - _adjusted_dem[mask]) ** 2)
        return mse

    # Use scipy.optimize.minimize to find the optimal constant value
    result = minimize(mse_difference, x0=[0])
    adjustment_value = result.x[0]

    # Debugging output
    print("Optimization Result:", result)
    print("Optimal Adjustment Value:", adjustment_value)

    # Adjust the other DEM with the found constant value
    adjusted_dem = other_dem + adjustment_value

    return adjusted_dem, adjustment_value
