import os
from yaml import load, Loader
import json
from langchain.chat_models import ChatOpenAI
from Arcadia.chain.chain import chain_factory
from Arcadia.answer.parser import CoTAnswerParser


if __name__ == '__main__':
    # 加载配置
    print('加载配置...')
    with open('config.yml') as f:
        config = load(f, Loader)

    # 设置网络代理
    print('设置网络代理...')
    os.environ['http_proxy'] = config['proxy']
    os.environ['https_proxy'] = config['proxy']

    # 初始化OpenAI接口
    print('初始化OpenAI接口...')
    openai = ChatOpenAI(
        model_name=config['openai']['model'],
        temperature=config['openai']['temperature'],
        openai_api_key=config['openai']['api-key']
        )

    # 初始化回答解析器
    print('初始化回答解析器...')
    cap = CoTAnswerParser()

    # 初始化提示生成器
    print('初始化提示生成器...')
    pg = chain_factory(openai, cap)

    # 读取数据
    print('读取数据...')
    assert os.path.exists(config['json']['time-series']), f"{config['json']['news']}不存在"
    assert os.path.exists(config['json']['news']), f"{config['json']['news']}不存在"
    with open(config['json']['time-series']) as f:
        ts_data = json.load(f)
    with open(config['json']['news']) as f:
        news_data = json.load(f)

    # 与ChatGPT交互
    print('与ChatGPT交互...')
    result = pg.run(
        timeseries=ts_data,
        news=news_data
    )
    
    # 保存结果
    print('保存结果...')
    with open(config['json']['result'], 'w') as f:
        json.dump(result, f, ensure_ascii=False)
