import pandas as pd
import numpy as np
from tqdm import tqdm
from .base import CSMAR
from .industry import CSMARIndustry
from data_provider.data import PriceData


def sector_average(sector_pivot_table):
    total_trade_share = sector_pivot_table['share'].apply(np.sum, axis=1, result_type='reduce')
    total_trade_value = sector_pivot_table['value'].apply(np.sum, axis=1, result_type='reduce')
    total_market_value = sector_pivot_table['market_value'].apply(np.sum, axis=1, result_type='reduce') * 1000
    avg_price = total_trade_value / total_trade_share
    avg_turnover = total_trade_value / total_market_value
    sector_df = pd.DataFrame({
        'price': avg_price,
        'turnover': avg_turnover
    })
    return sector_df


class CSMARPrice(CSMAR, PriceData):
    # 列名对应表
    column_name_map = {
        'Stkcd': 'company',
        'Trddt': 'date',
        'Opnprc': 'open',
        'Hiprc': 'high',
        'Loprc': 'low',
        'Clsprc': 'close',
        'Dnshrtrd': 'share',
        'Dnvaltrd': 'value',
        'Dsmvosd': 'market_value',
        'Trdsta': 'trade_state',
        'Markettype': 'market_type',
        'Adjprcnd': 'comparable_price'
    }
    
    def load_data(self, reload=False):
        if self.df is not None and not reload:
            return
        super().load_data(reload)
        self.df = self.df[(self.df['trade_state'] == 1) & self.df['market_type'].isin([1, 4, 16, 32])]
    
    def average_by_industry(self, industry: "CSMARIndustry") -> pd.DataFrame:
        assert self.company_contain(industry), 'some companies in industry are not in price'
        df = self.df.set_index('company')
        industry_df = industry.df.set_index('company')

        # 分行业
        print('split by industry...')
        industries = industry_df.groupby(by='industry')

        # 计算每个行业平均
        print('average by industry...')
        sectors = []
        for sector_name in tqdm(industries.groups.keys()):
            # 该行业所有股票
            sector_member = industries.get_group(sector_name).index
            sector_price_df = df.loc[df.index.isin(sector_member)].reset_index()
            # 转换为pivot table
            sector_price_pivot = sector_price_df.pivot(index='date', columns='company')
            # 计算行业平均
            sector_df = sector_average(sector_price_pivot)
            sector_df.insert(0, 'industry', sector_name)
            sector_df.reset_index()
            sectors.append(sector_df)
        
        return pd.concat(sectors, axis=0).reset_index()


if __name__ == '__main__':
    csmar = CSMARPrice(
        trade_file='dataset/stock.parquet.gzip',
    )
    data, mask = csmar.load_data()
    print(f'data: {data.shape} {data.nbytes/1024*2:.2f}M')
    print(f'mask: {mask.shape} {mask.nbytes/1024*2:.2f}M')
