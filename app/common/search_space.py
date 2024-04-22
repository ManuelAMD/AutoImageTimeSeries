from dataclasses import dataclass, field
from enum import Enum
import abc

#from AutoImageTimeSeries.app.common.search_space import SearchSpaceType

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
    BASE_ARCHITECTURE: tuple = field(default=('conv_lstm_2d', "conv_lstm_2d"), hash= False)

    CONV_LSTM_LAYERS_N_MIN: int = 1
    CONV_LSTM_LAYERS_N_MAX: int = 8
    CONV_LSTM_FILTERS_BASE_MULTIPLIER: int = 2
    CONV_LSTM_FILTERS_MIN: int = 1
    CONV_LSTM_FILTERS_MAX: int = 16
    CONV_LSTM_FILTERS_SIZES: tuple = (3,5,7)

    NORMALIZATION_LAYER: tuple = (True, False)
    WINDOW_SIZE_MIN: int = 4
    WINDOW_SIZE_MAX: int = 20

    CONV_2D_LAYERS_N_MIN: int = 0
    CONV_2D_LAYERS_N_MAX: int = 5
    CONV_2D_FILTERS_BASE_MUlTIPLIER: int = 4
    CONV_2D_FILTERS_MIN: int = 1
    CONV_2D_FILTERS_MAX: int = 16
    CONV_2D_FILTERS_SIZES: int = (3,5,7)
    
    @staticmethod
    def get_type() -> SearchSpaceType:
        return SearchSpaceType.IMAGE_TIME_SERIES
    
    @classmethod
    def get_hash(cls) -> str:
        return hash(cls())