import pandas as pd


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


__all__ = [
    to_pivot
]
