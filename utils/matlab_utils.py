#!/usr/bin/env python3

"""
Contains function equivalents for matlab in python
Project: ITIDE
Author: SCS
Date: 30.07.2021
"""

import numpy as np
import pandas as pd


def matlab_dot(a, b, axis):
    """
    Equivalent to Matlab's dot function.
    dot(a, b) !=^ np.dot(a, b) if both a and b are N-D arrays (where N >= 2),
    but e.g. dot(a, b, 1) =^ np.sum(a.conj() * b, axis=1).
    '*' is elementwise multiplication.
    Sum of each row (axis = 1) gives the dot product of the horizontal vectors of each line.
    @param a: Vector
    @param b: Vector
    @param axis: Compute matlab's dot product along this axis.
    @return: Matlab dot product.
    """
    ret = np.sum(a.conj() * b, axis=axis)
    return ret


def matlab_ismember(a, b, rows=False) -> np.ndarray:
    """
    Returns a boolean array of the same shape as el that is True where
    an element of el is in test_el and False otherwise.
    @param a: Input array
    @param b: Array against which to test each value of el.
    @param rows: If true only rows are compared and behavior is similar
    to matlabs ismember function.
    @return: Boolean array
    """
    a = np.array(a)
    b = np.array(b)
    if rows and ((a.ndim != 2) or (b.ndim != 2)):
        raise ValueError(f'Arrays must have two dimensions {(a.ndim, b.ndim)}')
    elif rows and (a.shape[1] != b.shape[1]):
        raise ValueError(f'Arrays must have the same number of columns to compare rows {(a.shape[1], b.shape[1])}')

    if rows:
        lia = np.isin(a, b).all(axis=1)
        cols = ['a', 'b', 'c']
        dfa = pd.DataFrame(a, columns=cols)
        dfb = pd.DataFrame(b, columns=cols)
        df_merge = dfa.reset_index().merge(dfb.reset_index(), on=cols)
        locb = df_merge.index_y.to_numpy()
        if (locb.ndim != 1) or (locb.size != a.shape[0]):
            raise ValueError(f'locb is malformed')
        return lia, locb
    else:
        lia = np.isin(a, b)
        return lia
