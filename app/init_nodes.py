from app.dataset_files.dataset import *
from app.master.optimization_job import OptimizationJob
from app.common.arquitecture_factory import *
from app.worker.worker_training import WorkerTraining
import json

class InitNodes:
    def master(self):
        config_file = input("Config file name (json): ")
        print(config_file)
        config_json = self.read_json_file(config_file)
        print(config_json)
        model_architecture_factory: ModelArchitectureFactory = self.get_model_architecture(config_json['arch_type'])
        dataset: Dataset = self.get_dataset_type(config_json)
        optimization = OptimizationJob(dataset, model_architecture_factory, config_json)
        optimization.start_optimization(trials= config_json['trials'])

    def worker(self):
        config_file = input("Config file name (json): ")
        print(config_file)
        config_json = self.read_json_file(config_file)
        print(config_json)
        model_architecture_factory: ModelArchitectureFactory = self.get_model_architecture(config_json['arch_type'])
        dataset: Dataset = self.get_dataset_type(config_json)
        worker_training = WorkerTraining(dataset, model_architecture_factory, config_json)
        worker_training.start_worker()

    def read_json_file(self, filename):
        f = open('configurations/{}'.format(filename), "r")
        parameters = json.load(f)
        print(type(parameters))
        return parameters
    
    def get_model_architecture(self, arch_type: int) -> ModelArchitectureFactory:
        if arch_type == 1:
            return ImageTimeSeriesArchitectureFactory()
    
    def get_dataset_type(self, parameters):
        if parameters['data_type'] == 'ITS':
            return ImageTimeSeriesDataset(parameters['name'], 
                                        (parameters['window_size'],
                                        parameters['rows'], 
                                        parameters['cols'], 
                                        parameters['channels']))
        