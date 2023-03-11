import pandas as pd
import numpy as np
from tqdm import tqdm
from .base import CSMAR
from data_provider.data import FutureDeliveryDateData


class CSMARFuture(CSMAR, FutureDeliveryDateData):
    column_name_map = {
        'delivery_date': 'delivery_date'
    }

    @property
    def date_list(self):
        return self.df['delivery_date'].to_list()
