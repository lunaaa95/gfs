import pandas as pd
import numpy as np
from tqdm import tqdm
from .base import CSMAR


class CSMARIndustry(CSMAR):
    # 列名对应表
    column_name_map = {
        'Stkcd': 'company',
        'Nnindnme': 'industry',
        'Markettype': 'market_type'
    }

