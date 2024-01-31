import pandas as pd
import numpy as np

def convert_to_latex(data):
    """
    Convert a pandas DataFrame, Series, or a NumPy array to a LaTeX table format.
    For DataFrame columns of integer type, format without decimal points.
    """
    latex_str = ""

    def format_value(val, is_int):
        """Format the value based on its type."""
        return str(int(val)) if is_int and not pd.isna(val) else str(val)

    # Check if the data is a DataFrame
    if isinstance(data, pd.DataFrame):
        int_cols = data.dtypes == 'int'
        for index, row in data.iterrows():
            row_str = " & ".join([format_value(val, int_cols[col]) for col, val in row.iteritems()])
            latex_str += f"{index} & {row_str} \\\\\n"

    # Check if the data is a Series
    elif isinstance(data, pd.Series):
        is_int = data.dtype == 'int'
        for index, value in data.items():
            formatted_value = format_value(value, is_int)
            latex_str += f"{index} & {formatted_value} \\\\\n"

    # Check if the data is a NumPy array
    elif isinstance(data, np.ndarray):
        for row in data:
            row_str = " & ".join(map(str, row))
            latex_str += row_str + " \\\\\n"

    else:
        raise TypeError("Input must be a pandas DataFrame, Series, or a NumPy array.")

    print(latex_str)

