import os
os.environ['http_proxy'] = 'http://10.177.27.237:7890'
os.environ['https_proxy'] = 'http://10.177.27.237:7890'

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from Arcadia.chain.chain import NewsPredictChain
from Arcadia.prompt.template import news_predict_prompt
api_key = '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'


if __name__ == '__main__':
    openai = OpenAI(model_name='gpt-3.5-turbo', temperature=0, openai_api_key=api_key)
    llm_chain = LLMChain(prompt=news_predict_prompt, llm=openai)

    npc = NewsPredictChain(chain=llm_chain)

    trend = '震荡'
    volatility = '低'
    active_level = '中等'
    news = '汉钟精机:涡旋压缩机明年逐渐量产.汉钟精机计划明年开始逐渐量产涡旋压缩机，首先推出空气产品，然后陆续推出制冷、冷冻等方面的产品，产能将根据市场需求而定。公司主要从事压缩机应用技术的研制开发、生产销售及售后服务。'

    # trend = '震荡'
    # volatility = '低'
    # active_level = '低'
    # news = '安信信托预计上半年净利润同比增长约65%.安信信托发布上半年业绩预告，预计净利润同比增长约65%，原因是期内业务收入增长。'

    test = npc.run(trend=trend, volatility=volatility, active_level=active_level, news=news)

    print(test)
