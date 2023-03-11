class PromptRule:
    states = {
        0: '好',
        1: '坏'
    }
    
    @classmethod
    def to_text(cls, state):
        return cls.states[state]
    
    @classmethod
    def to_state(cls, text):
        rules = {v: k for k, v in cls.states.items()}
        return rules[text]


class TrendRule(PromptRule):
    name = '价格'
    states = {
        0: '下跌',
        1: '上涨',
        2: '震荡'
    }


class VolatilityRule(PromptRule):
    name = '波动率'
    states = {
        0: '低',
        1: '高'
    }


class ActiveLevelRule(PromptRule):
    name = '交易活跃程度'
    states = {
        0: '低',
        1: '高',
        2: '中等'
    }


class DeliveryDayRule(PromptRule):
    states = {
        0: '不在股指期货交割日',
        1: '临近股指期货交割日'
    }


class HolidayRule(PromptRule):
    states = {
        0: '不在节假日',
        1: '临近节假日'
    }
