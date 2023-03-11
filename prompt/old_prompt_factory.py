from .rule import TrendRule, VolatilityRule, ActiveLevelRule, DeliveryDayRule, HolidayRule
from .state import get_trend_and_volatility_state, get_turnover_state, get_date_states
import pandas as pd
from csmar.price import CSMARPrice
from csmar import to_pivot
from chinese_calendar import get_holidays
from datetime import datetime as dt


class PromptFactory:
    prompt_rules = {
            'trend': TrendRule,
            'volatility': VolatilityRule,
            'active_level': ActiveLevelRule,
            'delivery_day': DeliveryDayRule,
            'holiday': HolidayRule
        }

    def __init__(self, stock_file, delivery_file,
    include_weekends=False, window_size=5) -> None:
        self.stock_file = stock_file
        self.delivery_file = delivery_file
        self.include_weekends = include_weekends
        self.window_size = window_size
    
    def _load_data(self):
        stock = CSMARPrice(self.stock_file)
        stock.load_data()
        
        price = pd.DataFrame(data={
            'date': pd.to_datetime(stock.df['date']),
            'price': stock.df['comparable_price'],
            'company': stock.df['company']})
        price, _ = to_pivot(price, x='date', y='company', fillna='ffill')
        self.stock_price = price.resample('B').mean().fillna(method='ffill')

        turnover = pd.DataFrame(data={
            'date': pd.to_datetime(stock.df['date']),
            'turnover': stock.df['value']/stock.df['market_value'],
            'company': stock.df['company']})
        turnover, _ = to_pivot(turnover, x='date', y='company', fillna='ffill')
        self.stock_active_level = turnover.resample('B').mean().fillna(method='ffill')
        
        self.delivery_date = pd.read_parquet(self.delivery_file)['delivery_date'].to_list()
        
        holidays = get_holidays(
            dt.strptime('2016-01-01', '%Y-%m-%d'),
            dt.strptime('2022-12-31', '%Y-%m-%d'),
            include_weekends=self.include_weekends)
        self.holiday_date = [d.strftime('%Y-%m-%d') for d in holidays]

        self.date = self.stock_price.index.strftime('%Y-%m-%d')[self.window_size:]
        self.company = self.stock_price.columns.get_level_values(1)
    
    def _get_states(self):
        price_series = self.stock_price.to_numpy()
        turnover_series = self.stock_active_level.to_numpy()
        
        ts, vs, _ = get_trend_and_volatility_state(
            price_series,
            window_size=self.window_size)
        tos = get_turnover_state(
            turnover_series,
            window_size=self.window_size
        )
        ds, hs = get_date_states(
            self.date,
            self.delivery_date,
            self.holiday_date
        )
        return ts, vs, tos, ds, hs
    
    def load(self):
        self._load_data()
        self.trend_state, self.volatility_state, self.turnover_state, self.delivery_state, self.holiday_state = self._get_states()

    @classmethod
    def build_one_prompt(cls, state_dict):
        """
        state_dict:
        trend, volatility, active_level, delivery_day, holiday
        """
        prompt = ''
        for name, state in state_dict.items():
            prompt += cls.prompt_rules[name].to_text(state)
        return prompt
    
    def __getitem__(self, idx):
        state_dict = {
            'trend': self.trend_state[idx],
            'volatility': self.volatility_state[idx],
            'active_level': self.turnover_state[idx],
            'delivery_day': self.delivery_state[idx[0]],
            'holiday': self.holiday_state[idx[0]]
        }
        return self.build_one_prompt(state_dict)
