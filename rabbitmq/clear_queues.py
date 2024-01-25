import asyncio
import aio_pika

async def main(loop, params=None):
    if params == None:
        connection = await aio_pika.connect_robust(
            "amqp://{}:{}@{}/".format("guest", "guest", "localhost"), loop=loop
        )
    else:
        connection = await aio_pika.connect_robust(
            "amqp://{}:{}@{}/".format(params['user'], params['password'], params['host']), loop=loop
        )
    queue_names = ["models", "trainedModels"]
    print("Trying connection")
    async with connection:
        print("Connected!")
        #Create communication channel
        channel = await connection.channel()

        for queue_name in queue_names:
            queue = await channel.declare_queue(
                queue_name, durable= True
            )
            await queue.purge()
    print("Queues cleared")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))
    loop.close()