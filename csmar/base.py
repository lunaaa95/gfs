import pandas as pd
import numpy as np
from tqdm import tqdm


"""
传入的CSMAR需要保证数据已加载
"""


class CSMAR:
    # 列名对应表
    column_name_map = {
        'Stkcd': 'company',
        'Trddt': 'date'
    }

    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.df = None
    
    def load_data(self, reload=False):
        if self.df is not None and not reload:
            return
        # 读取数据
        print(f'{self.__class__.__name__} load data...')
        raw_df = pd.read_parquet(self.file_path)

        # 重命名列
        raw_df = raw_df[self.__class__.column_name_map.keys()].rename(columns=self.__class__.column_name_map)

        self.df = raw_df
    
    def company_intersect(self, other: "CSMAR") -> "CSMAR":
        # 读取数据
        self.load_data()
        other_df = other.df.set_index('company')
        df = self.df.set_index('company')

        # 股票交集
        print(f'{self.__class__.__name__} filter data...')
        selected = df.index.intersection(other_df.index)

        # 筛选数据
        result = self.__class__(self.file_path)
        result.df = df.loc[df.index.isin(selected)].reset_index()

        return result
    
    def company_contain(self, other: "CSMAR") -> bool:
        # 读取数据
        self.load_data()
        other_df = other.df.set_index('company')
        df = self.df.set_index('company')

        return np.all(other_df.index.isin(df.index))
