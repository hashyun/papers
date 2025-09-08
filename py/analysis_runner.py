"""Utility functions to reduce repetition across EPU analysis notebooks.

This module centralizes common imports and provides a helper function
for running the same analysis across multiple datasets.
"""

from __future__ import annotations

from typing import Dict, Any

import pandas as pd

import granger_analysis


def run_epu_analysis(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Run a basic EPU analysis for multiple datasets.

    Parameters
    ----------
    dataframes:
        Mapping of dataset name to the corresponding :class:`pandas.DataFrame`.

    Returns
    -------
    dict
        Mapping of dataset name to the stationarity results produced by
        :func:`granger_analysis.check_stationarity_pro`.
    """
    results: Dict[str, Any] = {}
    for name, df in dataframes.items():
        results[name] = granger_analysis.check_stationarity_pro(df)
    return results
