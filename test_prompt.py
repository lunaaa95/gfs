from prompt.prompt import PromptFactory
from csmar.price import CSMARPrice
from csmar.future import CSMARFuture
from csmar.holiday import Holiday
from csmar.news import CSMARNews

from tqdm import tqdm
import pickle
import numpy as np


if __name__ == '__main__':
    price_data = CSMARPrice('dataset/stock.parquet.gzip')
    future_data = CSMARFuture('dataset/future.parquet.gzip')
    news_data = CSMARNews('dataset/event.parquet.gzip')
    holiday_data = Holiday()

    price_data.load_data()
    news_data.load_data()
    future_data.load_data()
    holiday_data.load_data()

    news_data = news_data.company_intersect(price_data)
    price_data = price_data.company_intersect(news_data)

    pf = PromptFactory(
        price_data=price_data,
        news_data=news_data,
        future_data=future_data,
        holiday_data=holiday_data
    )

    # gpt-3.5可能见过的
    mask = np.load('cache/news_mask.npy')
    imas = np.where(mask.flatten() > 0)[0]

    num_test = 100
    offset = 5000
    records = []

    for i in tqdm(imas):
        prompt, news_count, target, company, date = pf[int(i+offset)]
        if news_count > 0:
            records.append((prompt, news_count, target, company, date))
        if len(records) > num_test:
            break
    
    with open('samples_old.pkl', 'wb') as f:
        pickle.dump(records, f)
    

    # gpt-3.5没见过的
    date = pf.date
    imas_d = np.where(date >= '2022-01-01')[0][0]
    imas_2022 = imas[imas >= imas_d]
    offset = 5000
    num_test = 100
    records = []

    for i in tqdm(imas):
        prompt, news_count, target, company, date = pf[int(i+offset)]
        if news_count > 0:
            records.append((prompt, news_count, target, company, date))
        if len(records) > num_test:
            break
    
    with open('samples_2022.pkl', 'wb') as f:
        pickle.dump(records, f)

    pass
