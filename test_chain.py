import os
os.environ['http_proxy'] = 'http://10.177.27.237:7890'
os.environ['https_proxy'] = 'http://10.177.27.237:7890'

from langchain.llms import OpenAI
from Arcadia.chain.chain import chain_factory
from Arcadia.answer.parser import GeneralAnswerParser
from Arcadia.prompt.rule import TrendRule, VolatilityRule, ActiveLevelRule
api_key = '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'


if __name__ == '__main__':
    openai = OpenAI(model_name='gpt-3.5-turbo', temperature=0.3, openai_api_key=api_key)

    gap = GeneralAnswerParser([TrendRule, VolatilityRule, ActiveLevelRule])

    fpc = chain_factory(openai, gap)

    
    # trend = '震荡'
    # volatility = '低'
    # active_level = '中等'
    # title = '汉钟精机:涡旋压缩机明年逐渐量产'
    # body = """汉钟精机(002158)在近日披露的投资者关系活动记录中表示,市场上目前涡旋压缩机生产量已经很大,公司在细分产业中开发,生产无油压缩机,目前市场需求量不是很大。公司指出,涡旋压缩机第一步推出空气产品,陆续会推出制冷、冷冻等方面的产品。涡旋压缩机2017年开始逐渐量产,产能视市场需求而定。汉钟精机主要从事压缩机应用技术的研制开发、生产销售及售后服务。(责任编辑:DF120)"""


    trend = '下跌'
    volatility = '高'
    active_level = '高'
    title = '拓新药业:公司在核苷(酸)类原料药及医药中间体布局多年是国内最早专业从事核苷(酸)类产品的企业之一'
    body = """每经AI快讯,有投资者在投资者互动平台提问:请问真实生物为何选择与贵司共同合作研发阿兹夫定,作为一款国家1.1类创新药,该药备受国家重视,所以贵司有什么优势能够承担此次研发工作?谢谢。拓新药业(301089.SZ)1月6日在投资者互动平台表示,公司在核苷(酸)类原料药及医药中间体布局多年,是国内最早专业从事核苷(酸)类产品的企业之一,在该领域产品全、技术成熟,并不断创新。近年来,随着核苷(酸)类原料药及医药中间体应用领域的扩展,公司利用特有优势,把握市场机会,加快市场开拓及生产研发,力争把握市场机遇,实现新的突破。(文章来源:每日经济新闻)文章来源:每日经济新闻责任编辑:91"""


    result = fpc.run(
        trend=trend,
        volatility=volatility,
        active_level=active_level,
        title=title,
        body=body
        )
    
    print(result)
