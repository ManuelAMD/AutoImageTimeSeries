import asyncio
import json
from dataclasses import astuple
import aio_pika
from aio_pika import IncomingMessage
from app.rabbitmq_connections.rabbit_connection_params import RabbitConnectionParams

#Base class for RabbitMQ operations
class BaseRabbitMQClient:

    def __init__(self, params: RabbitConnectionParams, loop: asyncio.AbstractEventLoop):
        #Connection variables
        (
            self.port,
            self.model_parameter_queue,
            self.model_performance_queue,
            self.host_url,
            self.user,
            self.password,
            self.virtual_host
        ) = astuple(params)
        self.loop = loop
    
    #Async function for queue connection preparation
    async def prepare_queues(self):
        connection: aio_pika.RobustConnection = await self._create_connection()
        async with connection:
            channel = await connection.channel()
            print("** Initializing queues... **")
            await channel.declare_queue(self.model_parameter_queue, durable= True)
            await channel.declare_queue(self.model_performance_queue, durable= True)
            await asyncio.sleep(3)
            print("** Queues declared properly! **")
        
    async def publish(self, queue_name: str, message_body: dict, auto_close_connection = True) -> aio_pika.Connection:
        connection = await self._create_connection()
        message_body_json = json.dumps(message_body).encode()
        if auto_close_connection:
            async with connection:
                await self._run_publish(connection, queue_name, message_body_json)
        else:
            await self._run_publish(connection, queue_name, message_body_json)
        return connection
    
    async def listen(self, queue_name: str, callback, auto_close_connection = True) -> aio_pika.Connection:
        connection = await self._create_connection()
        async def on_result_recieved(message: IncomingMessage):
            body_json = message.body.decode()
            body_dict = json.loads(body_json)
            await callback(body_dict)
            await message.ack()
        if auto_close_connection:
            async with connection:
                await self._run_listener(connection, queue_name, on_result_recieved)
        else:
            await self._run_listener(connection, queue_name, on_result_recieved)
        return connection
    
    @staticmethod
    async def _run_publish(connection, queue_name, message_body_json):
        routing_key = queue_name
        channel = await connection.channel()
        await channel.default_exchange.publish(
            message = aio_pika.Message(
                body = message_body_json,
                content_type = 'application/json',
                content_encoding = 'utf-8',
            ),
            routing_key = routing_key
        )
    
    @staticmethod
    async def _run_listener(connection, queue_name, callback):
        routing_key = queue_name
        channel: aio_pika.Channel = await connection.channel()
        await channel.set_qos(prefetch_count = 1)
        queue: aio_pika.Queue = await channel.declare_queue(routing_key, durable= True)
        await queue.consume(callback, no_ack= False)

    async def _create_connection(self) -> aio_pika.RobustConnection:
        return await aio_pika.connect_robust("amqp://{}:{}@{}/".format(self.user, self.password, self.host_url), loop= self.loop)