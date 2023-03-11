from langchain import PromptTemplate


_template = """你是一位阅读过大量中文新闻的诚实的股票分析师,同时你会考虑股票的价格趋势,波动率和交易活跃程度.
股票的价格趋势包括:
- 下降
- 上涨
- 震荡

股票的波动率包括:
- 低
- 高

股票的交易活跃程度包括:
- 高
- 中等
- 低

记住当前股票的价格趋势是{trend},波动率是{volatility},交易活跃程度是{active_level},并且{delivery_day}、{holiday},再阅读新闻:{news}
"""

def template_factory():
    return PromptTemplate(
        input_variables=['trend', 'volatility', 'active_level', 'delivery_day', 'holiday', 'news'],
        template=_template,
    )


"""
例子如:
1. 当前股票价格趋势上涨,交易活跃程度中等,新闻利好股票,接下来可能股票价格趋势上涨,交易活跃程度高.
2. 当前股票价格趋势上涨,交易活跃程度高,新闻利好股票,接下来可能股票价格趋势震荡,交易活跃程度高.
3. 当前股票价格趋势震荡,交易活跃程度低,新闻利好股票,接下来可能股票价格趋势上涨,交易活跃程度高.
4. 当前股票价格趋势下跌,波动率低,交易活跃程度低,新闻利好股票,接下来可能股票价格趋势震荡,交易活跃程度中等.
5. 当前股票价格趋势震荡,波动率高,交易活跃程度低,新闻利好股票,接下来可能股票价格趋势下降,交易活跃程度低.
"""