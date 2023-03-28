import pandas as pd

from Arcadia.answer.parser import CoTAnswerParser


if __name__ == '__main__':
    answer = pd.read_csv('output/cot_answers_2022.csv')
    parser = CoTAnswerParser()

    splited = answer['raw_output'].map(parser.parse)
    pred = pd.DataFrame(splited.to_list())

    pred = pred.rename(columns={
        'trend': 'answer_trend',
        'volatility': 'answer_volatility',
        'active_level': 'answer_active_level'
    })

    for k in answer.columns[-3:]:
        answer[k] = pred[k]
    
    answer.to_csv('output/cot_answers_2022_reparse.csv')
    