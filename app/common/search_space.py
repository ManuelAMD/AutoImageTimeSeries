from dataclasses import dataclass, field
from enum import Enum
import abc

from AutoImageTimeSeries.app.common.search_space import SearchSpaceType

class SearchSpaceType(Enum):
    IMAGE_TIME_SERIES = 'image_time_series'

@dataclass(frozen=True)
class SearchSpace:
    @staticmethod
    @abc.abstractmethod
    def get_type() -> SearchSpaceType:
        pass

    @classmethod
    @abc.abstractmethod
    def get_hash(cls) -> str:
        pass

@dataclass(frozen=True)
class ConvLSTMSearchSpace(SearchSpace):
    BASE_ARCHITECTURE: tuple = field(default=('conv_lstm_2d'))

    CONV_LSTM_BLOCKS_N_MIN: int = 1
    CONV_LSTM_BLOCKS_N_MAX: int = 6
    CONV_LSTM_FILTERS_BASE_MULTIPLIER: int = 16
    CONV_LSTM_FILTERS_MIN: int = 1
    CONV_LSTM_FILTERS_MAX: int = 6
    CONV_LSTM_FILTERS_SIZES: tuple = (7,7)    
    
    @staticmethod
    def get_type() -> SearchSpaceType:
        return SearchSpaceType.IMAGE_TIME_SERIES
    
    @classmethod
    def get_hash(cls) -> str:
        return hash(cls())