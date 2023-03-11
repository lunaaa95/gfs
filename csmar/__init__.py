import pandas as pd
import numpy as np


def to_pivot(df: pd.DataFrame, x: str, y: str, fillna=None):
    print('convert format...')
    # 转换为pivot table
    pivot = df.pivot(index=x, columns=y)

    # 按日期排序
    print('sort by date...')
    pivot = pivot.sort_index()

    # 填na
    result_mask = pivot.notna()
    if fillna is not None:
        pivot = pivot.fillna(method=fillna)
    else:
        pivot = pivot.fillna(value=0)
    
    return pivot, result_mask


def to_numpy(df: pd.DataFrame, x: str, y: str, fillna=None):
    print('convert format...')
    # 转换为pivot table
    pivot = df.pivot(index=x, columns=y)

    # 按日期排序
    print('sort by date...')
    pivot = pivot.sort_index()

    # 填na
    result_mask = pivot.notna()
    if fillna is not None:
        pivot = pivot.fillna(method=fillna)
    else:
        pivot = pivot.fillna(value=0)

    # 数据写成 总天数*个体数*特征数
    ndays, nfeatures, nindividual = pivot.index.size, *pivot.columns.levshape
    result_array = pivot.to_numpy(dtype=float, copy=True, na_value=0).reshape(ndays, nfeatures, nindividual).transpose(0, 2, 1)
    result_mask = result_mask.to_numpy(dtype=np.int8, copy=True, na_value=0).reshape(ndays, nfeatures, nindividual).transpose(0, 2, 1)

    return result_array, result_mask

__all__ = [
    to_numpy
]