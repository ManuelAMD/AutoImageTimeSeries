from app.dataset_files.dataset import *
from app.master.optimization_job import OptimizationJob
import json

class InitNodes:
    def master(self):
        config_file = input("Config file name (json): ")
        print(config_file)
        config_json = self.read_json_file(config_file)
        dataset: Dataset = self.get_dataset_type(config_json)
        optimization = OptimizationJob(dataset)

    def read_json_file(self, filename):
        f = open('configurations/{}'.format(filename), "r")
        parameters = json.load(f)
        print(type(parameters))
        return parameters
    
    def get_dataset_type(self, parameters):
        if parameters['data_type'] == 'ITS':
            return ImageTimeSeriesDataset(parameters['name'], (parameters['rows'], parameters['cols'], parameters['channels']))
        