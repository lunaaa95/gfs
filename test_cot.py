import os
import numpy as np
from langchain.llms import OpenAI
from Arcadia.chain.chain import chain_factory
from Arcadia.answer.parser import CoTAnswerParser
import tiktoken
from time import sleep
from tqdm import tqdm
import csv

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


if __name__ == '__main__':
    api_key = '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'
    os.environ['http_proxy'] = 'http://10.177.27.237:7890'
    os.environ['https_proxy'] = 'http://10.177.27.237:7890'

    data_file = 'cache/samples_202211.pkl'
    data = np.load(data_file, allow_pickle=True)

    openai = OpenAI(model_name='gpt-3.5-turbo', temperature=0, openai_api_key=api_key)
    cap = CoTAnswerParser()
    fpc = chain_factory(openai, cap)

    with open('output/cot_answers_202211.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'company', 'date',
            'prompt_trend', 'prompt_volatility', 'prompt_active_level', 'news_title', 'news_body', 'news_count',
            'target_trend', 'target_volatility', 'target_active_level',
            'raw_output', 'answer_trend', 'answer_volatility', 'answer_active_level', 
            ])

    for prompt_info, news_count, target_info, company, date in tqdm(data):
        # 填写prompt
        trend = prompt_info['trend']
        volatility = prompt_info['volatility']
        active_level = prompt_info['active_level']
        title = prompt_info['title']
        body = prompt_info['body'].replace('\t', '')

        # icl with gpt-3.5 turbo
        result = fpc.run(
            trend=trend,
            volatility=volatility,
            active_level=active_level,
            title=title,
            body=body
            )
        raw_answer = result['raw_output']
        answer = result['prediction']
        

        with open('output/cot_answers_202211.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                company, date,
                trend, volatility, active_level, title, body, news_count,
                target_info['trend'], target_info['volatility'], target_info['active_level'],
                raw_answer, answer['trend'], answer['volatility'], answer['active_level']
                ])

        sleep(6)
