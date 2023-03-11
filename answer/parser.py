import re
import spacy


class NeatAnswerParser:
    pattern = re.compile('价格趋势是(?P<trend>\w+)，波动率是(?P<volatility>\w+)，交易活跃程度是(?P<active>\w+)。')
    keys = [
        'trend',
        'volatility',
        'active'
    ]
    
    @classmethod
    def parse(cls, answer_string):
        matches = cls.pattern.search(answer_string)
        return {k: matches.group(k) for k in cls.keys} if matches else None


class GeneralAnswerParser:
    """基于依存分析,解析gpt的回答
    """

    fck = {'有所', '但', '受到', '停牌'}

    def __init__(self, rules, trained_pipeline='zh_core_web_sm') -> None:
        self.nlp = spacy.load(trained_pipeline)
        self.trained_pipeline = trained_pipeline

        self.rules = rules
        self.attributes = {r.name: '|'.join(r.states.values()) for r in self.rules}
        self.re_patterns = {k: re.compile(f'{k}.*(?P<state>{self.attributes[k]})') for k in self.attributes.keys()}

        self.nlp.tokenizer.pkuseg_update_user_dict(self.attributes)
    
    def parse(self, answer_sentence):
        bowl = {}

        answer_sentence = answer_sentence.replace('：','')
        answer_sentence = answer_sentence.replace(':', '')
        answer_sentence = answer_sentence.replace(' ', '')

        answer_sentence = answer_sentence.replace('，', '\n')
        answer_sentence = answer_sentence.replace('。', '\n')

        all_sentences = []
        temp = answer_sentence.split('\n')
        for t in temp:
            if len(t) > 0:
                all_sentences.append(t)

        for sentence in all_sentences:
            doc = self.nlp(sentence)

            for token in doc:
                if token.text in self.attributes:

                    match = self.re_patterns[token.text].search(sentence)
                    if match is not None:
                        bowl[token.text] = match.group('state')
                        continue

                    root = None
                    for a in token.ancestors:
                        if a.tag_ == 'VV' and a.text not in self.__class__.fck:
                            root = a
                            break
                    if root is None:
                        bowl[token.text] = '无法判断'
                        continue

                    adv = ''
                    for c in root.children:
                        if c.tag_ == 'AD' and c.text not in self.__class__.fck:
                            adv += c.text
                    bowl[token.text] = adv + root.text
                
                if len(bowl) == len(self.rules):
                    return bowl
        
        return bowl
