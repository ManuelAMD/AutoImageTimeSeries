import time
import asyncio
import aio_pika
from dataclasses import asdict
from app.dataset_files.dataset import *
from app.common.arquitecture_factory import *
from app.master.optimization_strategy import *
from app.common.model_communication import *
from app.common.model import *
from app.rabbitmq_connections.rabbit_connection_params import RabbitConnectionParams
from app.rabbitmq_connections.master_rabbitmq_client import MasterRabbitMQClient
from app.rabbitmq_connections.rabbitmq_monitor import *

class OptimizationJob:
    def __init__(self, dataset: Dataset, model_architecture_search: ModelArchitectureFactory, parameters):
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.loop = asyncio.get_event_loop()
        print(dataset.get_tag())
        self.search_space: ModelArchitectureFactory = model_architecture_search
        self.dataset: Dataset = dataset
        self.optimization_strategy = OptimizationStrategy(self.search_space, self.dataset, parameters)
        rabbit_connection_params = RabbitConnectionParams.new(parameters)
        self.rabbitmq_client = MasterRabbitMQClient(rabbit_connection_params, self.loop)
        self.rabbitmq_monitor = RabbitMQMonitor(rabbit_connection_params)

    def start_optimization(self, trials: int):
        self.start_time = time.time()
        #Agregar, en caso de ser necesario, la conexión al maestro
        self.loop.run_until_complete(self.run_optimization_startup())
        connection = self.loop.run_until_complete(self.run_optimization_loop(trials))
        try:
            self.loop.run_forever()
        finally:
            self.loop.run_until_complete(connection.close())

    async def run_optimization_startup(self):
        print("** Running optimization startup **")
        await self.rabbitmq_client.prepare_queues()
        queue_status: QueueStatus = await self.rabbitmq_monitor.get_queue_status()
        for i in range (0, queue_status.consumer_count + 1):
            await self.generate_model()

    async def run_optimization_loop(self, trials: int) -> aio_pika.Connection:
        connection = await self.rabbitmq_client.listen_for_model_resutls(self.on_model_results)
        return connection
    
    async def on_model_results(self, response: dict):
        model_training_response = ModelTrainingResponse.from_dict(response)
        print("++ Recieved response ++")
        cad = "{} | {} | {}".format(model_training_response.id, model_training_response.performance, model_training_response.finished_epochs)
        print(cad)
        action: Action = self.optimization_strategy.report_model_response(model_training_response)
        print("++ Finished a model ++")
        print("++ The model got a {} value on the evaluation".format(self.optimization_strategy.get_training_total()))
        if action == Action.GENERATE_MODEL:
            await self.generate_model()
        elif action == Action.WAIT:
            print("** Waiting for the model **")
        elif action == Action.START_NEW_PHASE:
            queue_status: QueueStatus = await self.rabbitmq_monitor.get_queue_status()
            print("** New phase, deep training **")
            for i in range(0, queue_status.consumer_count + 1):
                await self.generate_model()
        elif action == Action.FINISH:
            print("** Finished training **")
            best_model = self.optimization_strategy.get_best_model()
            await self.log_results(best_model, self.optimization_strategy.get_best_exploration_ITS_model(), self.optimization_strategy.deep_training_models_completed)
            model = Model(best_model.model_training_request, self.dataset)
            model.is_model_valid()
            self.loop.stop()

    async def generate_model(self):
        print("-- Generating a new model --")
        model_training_request: ModelTrainingRequest = self.optimization_strategy.recommend_model()
        model = Model(model_training_request, self.dataset)
        if not model.is_model_valid():
            print("*** The model is not a valid one ***")
        else:
            await self.send_model_to_broker(model_training_request)
            print("*** The model was successfully generated ***")
        
    async def send_model_to_broker(self, model_training_request: ModelTrainingRequest):
        model_training_request_dict = asdict(model_training_request)
        print("Model trianing request", model_training_request_dict)
        await self.rabbitmq_client.publish_model_params(model_training_request_dict)

    async def log_results(self, best_model, best_exploration_model, next_models):
        print("-------------------------------------------------------------")
        print(next_models)
        print("Finishing process")
        filename = best_model.model_training_request.experiment_id
        f = open(filename, "a")
        model_info_json = json.dumps(asdict(best_model))
        f.write(model_info_json)
        f.close()

        print("Finished optimization")
        print("besto model: ")
        print(model_info_json)

        elapsed_seconds = time.time() - self.start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))

        time_text = "\n Optimization took: {} (hh:mm:ss), {} (Seconds)".format(elapsed_time, elapsed_seconds)
        print(time_text)

        f = open(filename, "a")
        f.write(time_text)
        f.close()

        f = open(filename, "a")
        model_info = json.dumps(asdict(best_exploration_model))
        f.write(model_info)
        f.close()

        f = open(filename, "a")
        for m in next_models[1:]:
            model_info = json.dumps(asdict(m))
            f.write(model_info)
        f.close()

        print("\n -------------------------------------------------------- \n")