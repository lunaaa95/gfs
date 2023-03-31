from langchain.chains.base import Chain
from langchain.chains import LLMChain
import tiktoken

from typing import List, Dict

from ..answer.parser import AnswerParser
from ..prompt.template import news_predict_prompt, news_summary_prompt
from ..prompt.series import TimeSeriesAnalysis


def chain_factory(llm, answer_parser):
    nsc = NewsSummaryChain(chain=LLMChain(prompt=news_summary_prompt, llm=llm))
    npc = NewsPredictChain(chain=LLMChain(prompt=news_predict_prompt, llm=llm))
    fpc = FullPredictionChain(news_summary_chain=nsc, news_predict_chain=npc, answer_parser=answer_parser)
    tsa = TimeSeriesAnalysis()
    pg = PromptGenerator(ts_analysis=tsa, full_chain=fpc)
    return pg


def num_tokens_from_string(string: str, encoding_name: str='cl100k_base') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


class NewsSummaryChain(Chain):
    chain: LLMChain

    @property
    def input_keys(self) -> List[str]:
        return ['title', 'body']
    
    @property
    def output_keys(self) -> List[str]:
        return ['summary']
    
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        paragraphs = inputs['body'].split('\n')
        preceding = inputs['title'] + '.'

        for i, p in enumerate(paragraphs):
            if num_tokens_from_string(p) > 3000:
                p = p.replace('.', '。')
                ps = p.split('。')
                paragraphs[i] = '。'.join(ps[len(ps)//2:])
                paragraphs.insert(i, '。'.join(ps[:len(ps)//2]))

        for p in paragraphs:
            p = p.replace('  ', '').replace('\t', '')
            preceding += self.chain.run(preceding=preceding, reading=p)
        return {'summary': preceding}


class NewsPredictChain(Chain):
    chain: LLMChain

    @property
    def input_keys(self) -> List[str]:
        return ['trend', 'volatility', 'active_level', 'news']
    
    @property
    def output_keys(self) -> List[str]:
        return ['raw_output']
    
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        raw_output = self.chain.run(inputs)
        return {'raw_output': raw_output}


class FullPredictionChain(Chain):
    news_summary_chain: NewsSummaryChain
    news_predict_chain: NewsPredictChain
    answer_parser: AnswerParser

    @property
    def input_keys(self) -> List[str]:
        return ['title', 'body', 'trend', 'volatility', 'active_level']

    @property
    def output_keys(self) -> List[str]:
        return ['output']
    
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        news = self.news_summary_chain.run({
            'title': inputs['title'],
            'body': inputs['body']
        })
        raw_output = self.news_predict_chain.run({
            'trend': inputs['trend'],
            'volatility': inputs['volatility'],
            'active_level': inputs['active_level'],
            'news': news
        })
        output = self.answer_parser.parse(raw_output)
        return {'output': {'raw_output': raw_output, 'prediction': output}}


class PromptGenerator(Chain):
    ts_analysis: TimeSeriesAnalysis
    full_chain: FullPredictionChain

    @property
    def input_keys(self) -> List[str]:
        return ['timeseries', 'news']
    
    @property
    def output_keys(self) -> List[str]:
        return ['output']
    
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        trend, volatility, active_level = self.ts_analysis(inputs['timeseries'])
        title = inputs['news']['title']
        body = inputs['news']['body']
        output = self.full_chain.run(
            trend=trend,
            volatility=volatility,
            active_level=active_level,
            title=title,
            body=body
        )
        return {'output': output}
