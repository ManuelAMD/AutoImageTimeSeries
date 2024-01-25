from dataclasses import dataclass

#connection params for RabbitMQ
@dataclass(frozen= True)
class RabbitConnectionParams:
    port: int
    model_parameter_queue: str
    model_performance_queue: str
    host_url: str
    user: str
    password: str
    virtual_host: str

    @staticmethod
    def new(params: dict):
        return RabbitConnectionParams(
            port = int(params['port']),
            model_parameter_queue = params['queue_publish'],
            model_performance_queue = params['queue_results'],
            host_url = params['host'],
            user = params['user'],
            password = params['password'],
            virtual_host = params['virtual_host']
        )