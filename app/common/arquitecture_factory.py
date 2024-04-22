import abc
from app.common.search_space import *
from app.common.arquitecture_parameters import *
import optuna

class ModelArchitectureFactory(abc.ABC):

    @abc.abstractmethod
    def generate_model_params(self):
        pass

    @abc.abstractmethod
    def get_search_space(self) -> SearchSpace:
        pass

class ImageTimeSeriesArchitectureFactory(ModelArchitectureFactory):
    search_space = ConvLSTMSearchSpace()
    
    def generate_model_params(self, recommender: optuna.Trial, input_dim: tuple):
        model_params: ImageTimeSeriesArchitectureParameters = ImageTimeSeriesArchitectureParameters.new()
        #Obtener la arquitectura base del optimizador
        #Base architecture by optuna
        model_params.base_architecture = recommender.suggest_categorical('BASE_ARCHITECTURE', self.search_space.BASE_ARCHITECTURE)
        if model_params.base_architecture == 'conv_lstm_2d':
            model_params = self._generate_conv_lstm_based_architecture(input_dim, recommender, model_params)
        model_params.window_size = recommender.suggest_int("WINDOW_SIZE", self.search_space.WINDOW_SIZE_MIN, self.search_space.WINDOW_SIZE_MAX)
        print("-- Architecture parameters --")
        print(recommender.params)
        return model_params
    
    def _generate_conv_lstm_based_architecture(self, input_dim: tuple, recommender: optuna.Trial, model_params: ImageTimeSeriesArchitectureParameters) -> ImageTimeSeriesArchitectureParameters:
        model_params.conv_lstm_layers_n = recommender.suggest_int("CONV_LSTM_LAYERS_N", self.search_space.CONV_LSTM_LAYERS_N_MIN, self.search_space.CONV_LSTM_LAYERS_N_MAX)
        for n in range (0, model_params.conv_lstm_layers_n):
            tag = "CONV_LSTM_LAYERS_FILTERS_{}".format(n)
            #Checar esta función
            filters = round(recommender.suggest_loguniform(tag, self.search_space.CONV_LSTM_FILTERS_MIN, self.search_space.CONV_LSTM_FILTERS_MAX))
            filters = filters * self.search_space.CONV_LSTM_FILTERS_BASE_MULTIPLIER
            model_params.conv_lstm_filters.append(filters)
            size = recommender.suggest_categorical("CONV_LSTM_LAYERS_FILTERS_SIZE_{}".format(n), self.search_space.CONV_LSTM_FILTERS_SIZES)
            model_params.conv_lstm_filters_sizes.append((size, size))
            
            #Checar si hay problema por si se tiene una capa de normalizaciòn como última de esta parte de la arquitectura
            normalized = recommender.suggest_categorical('NORMALIZATION_LAYER_{}'.format(n), self.search_space.NORMALIZATION_LAYER)
            model_params.normalization_layers.append(normalized)
        
        
        #model_params = self._generate_cnn_layer(recommender, model_params, input_dim[2])
        return model_params
    
    def _generate_cnn_layer(self, recommender: optuna.Trial, model_params: ImageTimeSeriesArchitectureParameters, channels: int) -> ImageTimeSeriesArchitectureParameters:
        model_params.cnn_layers_n = recommender.suggest_int("CNN_LAYERS_N", self.search_space.CONV_2D_LAYERS_N_MIN, self.search_space.CONV_2D_LAYERS_N_MAX)
        for n in range (0, model_params.cnn_layers_n):
            tag = "CNN_LAYERS_FILTERS_{}".format(n)
            filters = round(recommender.suggest_loguniform(tag, self.search_space.CONV_2D_FILTERS_MIN, self.search_space.CONV_2D_FILTERS_MAX))
            filters = filters * self.search_space.CONV_2D_FILTERS_BASE_MUlTIPLIER
            model_params.cnn_filters.append(filters)
            size = recommender.suggest_categorical("CNN_LAYERS_FILTERS_SIZE_{}".format(n), self.search_space.CONV_2D_FILTERS_SIZES)
            model_params.cnn_filters_size.append((size, size))
        #Output layer
        #model_params.cnn_filters_size.append("CNN_LAYERS_FILTERS_SIZE_{}".format(n), self.search_space.CONV_2D_FILTERS_SIZES)
        return model_params
    
    def get_search_space(self) -> SearchSpace:
        return self.search_space