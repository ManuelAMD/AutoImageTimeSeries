import asyncio
import time
import concurrent
import random
import logging
from dataclasses import asdict
import aio_pika
from app.common.model import Model
from app.common.model_communication import *
from app.rabbitmq_connections.rabbit_connection_params import RabbitConnectionParams
from app.common.search_space import *
from app.common.arquitecture_factory import *
from app.rabbitmq_connections.worker_rabbitmq_client import SlaveRabbitMQClient
from app.dataset_files.dataset import *

class WorkerTraining:
    def __init__(self, dataset: Dataset, model_architecture_factory: ModelArchitectureFactory, parameters):
        self.loop = asyncio.get_event_loop()
        self.dataset = dataset
        self.parameters = parameters
        rabbit_connection_params = RabbitConnectionParams.new(parameters)
        self.rabbitmq_client = SlaveRabbitMQClient(rabbit_connection_params, self.loop)
        self.search_space_hash = model_architecture_factory.get_search_space().get_hash()
        print("** Hash value: {} **".format(self.search_space_hash))
        self.model_type = model_architecture_factory.get_search_space().get_type()

    def start_worker(self):
        connection = self.loop.run_until_complete(self.start_listening())
        print("** Stop listening **")
        try:
            self.loop.run_forever()
        finally:
            self.loop.run_until_complete(connection.close())

    @staticmethod
    def fake_blocking_training():
        #Method for testing broker connection timeout
        for i in range(0, 10):
            time.sleep(1)
            print(i)
        res = random.uniform(0, 1)
        return res
    
    @staticmethod
    def train_model(info_dict: dict) -> float:
        dataset = info_dict['dataset']
        model_training_request = info_dict['model_request']
        dataset.load()
        model = Model(model_training_request, dataset)
        if info_dict['train_gpu'] == 1:
            return model.build_and_train()
        else:
            return model.build_and_train_cpu()
        
    async def start_listening(self) -> aio_pika.Connection:
        print("** Worker started! **")
        return await self.rabbitmq_client.listen_for_model_params(self.on_model_params_received)

    async def on_model_params_received(self, model_params):
        print("** Received model training request **")
        self.model_type = int(model_params['training_type'])
        model_training_request = ModelTrainingRequest.from_dict(model_params, self.model_type)
        if not self.search_space_hash == model_training_request.search_space_hash:
            raise Exception("Search space of master is different to this worker's search space")
        info_dict = {
            'dataset': self.dataset,
            'model_request': model_training_request,
            'train_gpu': self.parameters['train_gpu']
        }
        with concurrent.futures.ProcessPoolExecutor() as pool:
            training_val, did_finish_epochs = await self.loop.run_in_executor(pool, self.train_model, info_dict)
        #training_val = WorkerTraining.fake_blocking_training()
        did_finish_epochs = True
        model_training_response = ModelTrainingResponse(id = model_training_request.id, performance = training_val, finished_epochs = did_finish_epochs)
        await self.send_performance_to_broker(model_training_response)

    async def send_performance_to_broker(self, model_training_response: ModelTrainingResponse):
        print(model_training_response)
        model_training_response_dict = asdict(model_training_response)
        print(model_training_response_dict)
        await self.rabbitmq_client.publish_model_performance(model_training_response_dict)

def handle_exception(loop, context):
    msg = context.get("exception", context["message"])
    logging.error(f"caught exception: {msg}")
    logging.error(context["exception"])
    logging.info("** Shutting down... **")
    loop.stop()