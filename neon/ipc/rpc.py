"""
This file defines server and client classes for performing RPC.
Any number of RPC servers can serve requests on a given RPC queue.
Any number of RPC clients can send requests to a given RPC queue.
Load balancing about the existing servers on a queue happens automatically.
If a server dies during a computation, another server will finish the computation.

We don't presently have persistence guarantees for the messages, but that can
    be added without a huge amount of extra effort.

We should also deal with timeouts in a smarter way in the future.
"""

import pika
import uuid


class RpcServer(object):
    def __init__(self,rpc_queue, func):

        connection = pika.BlockingConnection(pika.ConnectionParameters(
                                    host='localhost'))

        self.func = func

        channel = connection.channel()
        channel.queue_declare(queue=rpc_queue)
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(self.on_request, queue=rpc_queue)

        print " Serving requests"
        channel.start_consuming()

    def on_request(self,ch,method,props,body):

        print " computing func(%s)" % (body,)
        response = self.func(body)

        ch.basic_publish(exchange='',
                         routing_key=props.reply_to,
                         properties=pika.BasicProperties(
                            correlation_id=props.correlation_id),
                         body=str(response))

        ch.basic_ack(delivery_tag = method.delivery_tag)


class RpcClient(object):
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
        return self.response

