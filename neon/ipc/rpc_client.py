import pika
import uuid

class NeonRpcClient(object):
    def __init__(self,rpc_queue):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(
                host='localhost'))

        self.rpc_queue = rpc_queue
        self.channel = self.connection.channel()

        result = self.channel.queue_declare(exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(self.on_response, no_ack=True,
                                   queue=self.callback_queue)

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, n):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(exchange='',
                                   routing_key=self.rpc_queue,
                                   properties=pika.BasicProperties(
                                         reply_to = self.callback_queue,
                                         correlation_id = self.corr_id,
                                         ),
                                   body=str(n))
        while self.response is None:
            self.connection.process_data_events()
        return int(self.response)

neon_rpc = NeonRpcClient('rpc_queue')

arg = int(raw_input("Give an integer to square: "))

print " [x] Making request"
response = neon_rpc.call(arg)
print " [.] Got %r" % (response,)
