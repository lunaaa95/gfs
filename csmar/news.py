import pandas as pd
import numpy as np
from tqdm import tqdm
from .base import CSMAR
import unicodedata as ucd
from data_provider.data import NewsData


class CSMARNews(CSMAR, NewsData):
    # 列名对应表
    column_name_map = {
        'Symbol': 'company',
        'DeclareDate': 'date',
        'Title': 'title',
        'NewsContent': 'body'
    }

    def load_data(self, reload=False):
        if self.df is not None and not reload:
            return
        super().load_data(reload)
        # 筛选有标签的新闻
        print('reserve news with company...')
        self.df = self.df[pd.notna(self.df['company'])]

        # 去除空格
        print('clean news title...')
        self.df['title'] = self.df['title'].map(lambda x: ucd.normalize('NFKC', x).replace(' ', ''))
        self.df['title'] = self.df['title'].str.replace(r'\t', '', regex=False)
        print('clean news body...')
        self.df['body'] = self.df['body'].map(lambda x: ucd.normalize('NFKC', x).replace(' ', ''))
        self.df['body'] = self.df['body'].str.replace(r'\t', '', regex=False)

        # 修改日期格式
        print('convert date format...')
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['date'] = self.df['date'].map(lambda x: x.date().strftime('%Y-%m-%d'))

    def gather_by_industry(self, industry):
        assert self.company_contain(industry), 'some companies in industry are not in news'
        df = self.df.set_index('company')
        industry_df = industry.df.set_index('company')

        # 分行业
        print('split by industry...')
        df = df.join(industry_df).reset_index()

        return df
