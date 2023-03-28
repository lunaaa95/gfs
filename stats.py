import pandas as pd
from tqdm import trange
import numpy as np


ugly = {
    '下降':0,
    '上涨':1,
    '中等':2,
    '略微上升':1,
    '略微下降':1,
    '震荡':2,
    '无法确定':2,
    '无法准确预测':2,
    '增加':1,
    '高':1,
    '无法判断':2,
    '低':0,
    '不确定':2,
    '无法进行':2,
    '不确定':2,
    '难以确定':2,
    '下跌':0
}
polar = ugly


if __name__ == '__main__':
    answer = pd.read_csv('simple_answers_2022_1.csv')

    target = answer[['target_trend', 'target_volatility', 'target_active_level']]
    target = target.applymap(lambda x: polar[x]).to_numpy()

    splited = answer['answer'].map(parser.parse)
    pred = pd.DataFrame(splited.to_list())
    pred = pred.applymap(lambda x: polar[x]).to_numpy()
    
    sd = set()
    for ss in splited.to_list():
        sf = set(ss.values())
        sd |= sf

    box = np.equal(target, pred)
    unsure = (pred > 1)

    for i in range(3):
        b = box[:,i]
        u = unsure[:,i]
        bu = b[u]
        print(i, bu.sum(), bu.sum()/len(bu))
    
    print(box.sum(axis=0) / box.shape[0])

    pass
