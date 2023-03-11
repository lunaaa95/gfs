from abc import ABCMeta, abstractmethod


class Data(metaclass=ABCMeta):
    @abstractmethod
    def load_data(self):
        pass


class DateData(Data, metaclass=ABCMeta):
    @property
    @abstractmethod
    def date_list(self) -> list:
        pass


class PriceData(Data, metaclass=ABCMeta):
    pass


class NewsData(Data, metaclass=ABCMeta):
    pass


class FutureDeliveryDateData(DateData, metaclass=ABCMeta):
    pass


class HolidayDateData(DateData, metaclass=ABCMeta):
    pass
