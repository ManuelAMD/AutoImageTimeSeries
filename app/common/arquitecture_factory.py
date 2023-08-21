import abc
from app.common.search_space import *

class ModelArchitectureFactory(abc.ABC):

    @abc.abstractmethod
    def generate_model_params(self):
        pass

    @abc.abstractmethod
    def get_search_space(self) -> SearchSpace:
        pass