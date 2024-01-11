import time
from app.dataset_files.dataset import *
from app.common.arquitecture_factory import *
from app.master.optimization_strategy import *
from app.common.model_communication import *
from app.common.model import *

class OptimizationJob:
    def __init__(self, dataset: Dataset, model_architecture_search: ModelArchitectureFactory, parameters):
        print(dataset.get_tag())
        self.search_space: ModelArchitectureFactory = model_architecture_search
        self.dataset: Dataset = dataset
        self.optimization_strategy = OptimizationStrategy(self.search_space, self.dataset, parameters)

    def start_optimization(self, trials: int):
        self.start_time = time.time()
        #Agregar, en caso de ser necesario, la conexión al maestro
        self.run_optimization_startup()
    
    def run_optimization_startup(self):
        print("** Running optimization startup **")
        #Aquí se preparan las colas
        # Se verifican las conexiones de rabbitmq
        # Por cada consumidor se debería generar un modelo.
        self.generate_model()

    def generate_model(self):
        print("-- Generating a new model --")
        model_training_request: ModelTrainingRequest = self.optimization_strategy.recommend_model()
        model = Model(model_training_request, self.dataset)
        if not model.is_model_valid():
            print("*** The model is not a valid one ***")
        else:
            print("*** The model was successfully generated ***")
        
    