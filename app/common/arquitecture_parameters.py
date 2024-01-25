from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin
from typing import List

@dataclass
class ModelArchitectureParameters(DataClassJsonMixin):

    @staticmethod
    def new():
        pass

@dataclass
class ImageTimeSeriesArchitectureParameters(ModelArchitectureParameters):
    base_architecture: str

    #Conv-LSTM-2D base architecture
    #Number of layers
    conv_lstm_layers_n: int
    #Per layer parameters
    conv_lstm_filters: List[int]
    conv_lstm_filters_sizes: List[tuple[int, int]]
    
    normalization_layers: List[bool]

    cnn_layers_n: int
    cnn_filters: List[int]
    cnn_filters_size: List[tuple[int, int]]

    @staticmethod
    def new():
        return ImageTimeSeriesArchitectureParameters(
            base_architecture = None,
            conv_lstm_layers_n = 0,
            conv_lstm_filters = list(),
            conv_lstm_filters_sizes = list(),
            normalization_layers = list(),
            cnn_layers_n = 0,
            cnn_filters = list(),
            cnn_filters_size = list()
        )