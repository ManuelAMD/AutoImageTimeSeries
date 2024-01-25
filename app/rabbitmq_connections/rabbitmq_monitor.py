from aiohttp import BasicAuth, ClientSession
from dataclasses import dataclass
from app.rabbitmq_connections.rabbit_connection_params import RabbitConnectionParams

@dataclass
class QueueStatus:
    queue_name: str
    consumer_count: int
    message_count: int

class RabbitMQMonitor(object):

    def __init__(self, params: RabbitConnectionParams):
        self.cp = params
        self.auth = BasicAuth(login = self.cp.user, password = self.cp.password)
    
    async def get_queue_status(self) -> QueueStatus:
        print("** Requesting queue status... **")
        async with ClientSession(auth = self.auth) as session:
            if self.cp.host_url == 'localhost':
                url = 'http://localhost:15672/api/queues/%2F/'
            else:
                url = "http://{}:{}/api/queues/%2F/{}".format(self.cp.host_url, self.cp.port, self.cp.model_parameter_queue)
            print(url)
            async with session.get(url) as resp:
                print(resp.status)
                #print(resp)
                body = await resp.json()
                #print(body[0])
                consumer_count = body[0]['consumers']
                message_count = body[0]['messages']
                queue_name = body[0]['name']
                queue_status = QueueStatus(
                    queue_name= queue_name,
                    consumer_count= consumer_count,
                    message_count= message_count
                )
                print("** Recieved queue status! **")
                cad = "{} | {} | {}".format(queue_status.queue_name, queue_status.consumer_count, queue_status.message_count)
                print(cad)
                return queue_status