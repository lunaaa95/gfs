from ..data import PriceData, NewsData, FutureDeliveryDateData, HolidayDateData
from .. import to_pivot

from Arcadia.prompt.rule import TrendRule, VolatilityRule, ActiveLevelRule, DeliveryDayRule, HolidayRule
from .state import get_trend_and_volatility_state, get_turnover_state, get_date_states

import pandas as pd
import numpy as np
from chinese_calendar import get_holidays
from datetime import datetime as dt


class PromptInformation:
    prompt_rules = {
            'trend': TrendRule,
            'volatility': VolatilityRule,
            'active_level': ActiveLevelRule,
            'delivery_day': DeliveryDayRule,
            'holiday': HolidayRule
        }
    
    def __init__(self, price_data: PriceData, news_data: NewsData,
                 future_data: FutureDeliveryDateData, holiday_data: HolidayDateData,
                 window_size=5, multiple_news=False) -> None:
        self.price_data = price_data
        self.news_data = news_data
        self.future_data = future_data
        self.holiday_data = holiday_data
        self.window_size = window_size
        self.multiple_news = multiple_news

        self._prepare()
        self.trend_state, self.volatility_state, self.turnover_state, self.delivery_state, self.holiday_state = self._get_states()
    
    def _prepare(self):
        price = pd.DataFrame(data={
            'date': pd.to_datetime(self.price_data.df['date']),
            'price': self.price_data.df['comparable_price'],
            'company': self.price_data.df['company']})
        price, _ = to_pivot(price, x='date', y='company', fillna='ffill')
        self.stock_price = price.resample('B').mean().fillna(method='ffill')

        turnover = pd.DataFrame(data={
            'date': pd.to_datetime(self.price_data.df['date']),
            'turnover': self.price_data.df['value']/self.price_data.df['market_value'],
            'company': self.price_data.df['company']})
        turnover, _ = to_pivot(turnover, x='date', y='company', fillna='ffill')
        self.stock_active_level = turnover.resample('B').mean().fillna(method='ffill')

        self.date = self.stock_price.index.strftime('%Y-%m-%d')[self.window_size:]
        self.company = self.stock_price.columns.get_level_values(1)

        self.delivery_date = self.future_data.date_list
        self.holiday_date = self.holiday_data.date_list

        news_df = pd.DataFrame(data={
            'date': pd.to_datetime(self.news_data.df['date']),
            'company':self.news_data.df['company'],
            'title': self.news_data.df['title'],
            'body': self.news_data.df['body']}).set_index('date')
        self.news_group = news_df.groupby(by='company')
    
    def _get_states(self):
        price_series = self.stock_price.to_numpy()
        turnover_series = self.stock_active_level.to_numpy()
        
        ts, vs, _ = get_trend_and_volatility_state(
            price_series,
            window_size=self.window_size)
        tos = get_turnover_state(
            turnover_series,
            window_size=self.window_size)
        ds, hs = get_date_states(
            self.date,
            self.delivery_date,
            self.holiday_date)
        
        return ts.flatten(), vs.flatten(), tos.flatten(), ds, hs
    
    def _get_info_at(self, idx):
        state_dict = {
            'trend': self.trend_state[idx],
            'volatility': self.volatility_state[idx],
            'active_level': self.turnover_state[idx],
            'delivery_day': self.delivery_state[idx//len(self.company)],
            'holiday': self.holiday_state[idx//len(self.company)]
        }
        info_dict = {name: self.prompt_rules[name].to_text(state) for name, state in state_dict.items()}
        return info_dict

    def _get_news_at(self, idx):
        """返回idx对应的候选新闻中的首条和全部候选的数量
        """
        company = self.company[idx%len(self.company)]
        news_df = self.news_group.get_group(company)
        last_date = self.date[idx//len(self.company) - 1] if idx >= len(self.company) else '1960-01-01'
        date = self.date[idx//len(self.company)]

        selected = news_df.loc[last_date: date]
        if not self.multiple_news:
            title = selected.iloc[0]['title'] if len(selected) > 0 else '没有新闻'
            body = selected.iloc[0]['body'] if len(selected) > 0 else '没有新闻.'
        else:
            pass  #TODO
        return title, body, len(selected), company, date


    def __getitem__(self, idx):
        assert isinstance(idx, int|np.int64) and idx < len(self)
        
        # 当前市场状态
        info_dict = self._get_info_at(idx)
        title, body, num_candi_news, company, date = self._get_news_at(idx)
        info_dict['title'] = title
        info_dict['body'] = body

        # 需要预测的市场状态
        target_dict = self._get_info_at(idx+self.window_size*len(self.company))
        
        return info_dict, num_candi_news, target_dict, company, date

    def __len__(self):
        return len(self.trend_state) - self.window_size*len(self.company)


__all__ = [
    PromptInformation
]
