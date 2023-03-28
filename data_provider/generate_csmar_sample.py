from .prompt import PromptInformation
from csmar.price import CSMARPrice
from csmar.future import CSMARFuture
from csmar.holiday import Holiday
from csmar.news import CSMARNews

from tqdm import trange
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

    pf = PromptInformation(
        price_data=price_data,
        news_data=news_data,
        future_data=future_data,
        holiday_data=holiday_data
    )

    # gpt-3.5可能见过的
    mask = np.load('cache/news_mask.npy')
    imas = np.where(mask.flatten() > 0)[0]

    # num_test = 100
    # offset = 5000
    # records = []

    # for i in tqdm(imas):
    #     prompt_info, news_count, target_info, company, date = pf[int(i+offset)]
    #     if news_count > 0:
    #         records.append((prompt_info, news_count, target_info, company, date))
    #     if len(records) > num_test:
    #         break
    
    # with open('cache/samples_old.pkl', 'wb') as f:
    #     pickle.dump(records, f)
    

    # gpt-3.5没见过的
    date = pf.date
    imas_d = np.where(date >= '2022-11-15')[0][0] * mask.shape[1]
    imas_2022 = imas[imas >= imas_d]
    print(len(imas_2022))
    offset = 0
    stride = 10
    num_test = 200
    records = []

    for nt in trange(num_test):
        i = imas_2022[int(nt*stride+offset)]
        print(i, len(pf))
        prompt_info, news_count, target_info, company, date = pf[i]
        if news_count > 0:
            records.append((prompt_info, news_count, target_info, company, date))
        # if len(records) > num_test:
        #     break
    
    with open('cache/samples_202212.pkl', 'wb') as f:
        pickle.dump(records, f)

    pass
