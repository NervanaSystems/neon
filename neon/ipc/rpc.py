# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
This file defines server and client classes for performing RPC.
Any number of RPC servers can serve requests on a given RPC queue.
Any number of RPC clients can send requests to a given RPC queue.
Load balancing about the existing servers on a queue happens automatically.
If a server dies during a computation, another server will finish the
computation.

We don't presently have persistence guarantees for the messages, but that can
    be added without a huge amount of extra effort.

We should also deal with timeouts in a smarter way in the future.
"""

import pika
import uuid


class RpcServer(object):
    def __init__(self, rpc_queue, func):

        # declare a blocking connection to rabbit server on localhost
        connection = pika.BlockingConnection(pika.ConnectionParameters(
            host='localhost'))

        self.func = func

        # declare (or open) a message queue listening for rpc calls
        channel = connection.channel()
        channel.queue_declare(queue=rpc_queue)

        # load balance the rpc calls about the servers on the channel
        channel.basic_qos(prefetch_count=1)

        # respond to requests on the specified channel with on_request below
        channel.basic_consume(self.on_request, queue=rpc_queue)

        print " Serving requests"
        channel.start_consuming()

    def on_request(self, ch, method, props, body):

        print " computing func(%s)" % (body,)
        response = self.func(body)

        # respond with function result
        ch.basic_publish(exchange='',  # use the default exchange
                         routing_key=props.reply_to,  # reply to correct client
                         properties=pika.BasicProperties(
                             # allows client to associate result with call
                             correlation_id=props.correlation_id),
                         body=str(response))

        # assert that response has been computed and sent
        # in principle, server could die after response but before ack
        # so the client needs to deal with that
        ch.basic_ack(delivery_tag=method.delivery_tag)


class RpcClient(object):
    def __init__(self, rpc_queue):

        self.connection = pika.BlockingConnection(pika.ConnectionParameters(
            host='localhost'))

        self.rpc_queue = rpc_queue
        self.channel = self.connection.channel()

        # declare a queue just for responses to this client
        result = self.channel.queue_declare(exclusive=True)
        self.callback_queue = result.method.queue

        # consume responses on that queue without acknowledging
        self.channel.basic_consume(self.on_response, no_ack=True,
                                   queue=self.callback_queue)

    def on_response(self, ch, method, props, body):
        # ignore responses without the proper correlation id
        # if we want to send multiple requests without getting a response
        # this will need to be slightly modified to maintain a map
        #   of outstanding correlation IDs or something
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, n):
        self.response = None
        # generate a UUID for this call
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(exchange='',
                                   routing_key=self.rpc_queue,
                                   properties=pika.BasicProperties(
                                       # tell server to reply to this client
                                       reply_to=self.callback_queue,
                                       # tell server the id of the call
                                       correlation_id=self.corr_id),
                                   body=str(n))

        while self.response is None:
            # continue processing responses on our exclusive channel
            # until we get the one that we want
            self.connection.process_data_events()
        return self.response
