import pandas as pd
import numpy as np
from tqdm import tqdm
from data_provider.data import HolidayDateData
from chinese_calendar import get_holidays
from datetime import datetime as dt


class Holiday(HolidayDateData):
    def __init__(self) -> None:
        pass
    
    def load_data(self):
        holidays = get_holidays(
            dt.strptime('2016-01-01', '%Y-%m-%d'),
            dt.strptime('2022-12-31', '%Y-%m-%d'),
            include_weekends=False)
        self.holiday_date = [d.strftime('%Y-%m-%d') for d in holidays]

    @property
    def date_list(self):
        return self.holiday_date
