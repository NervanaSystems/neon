import pika

def square(n):
    return n * n

class NeonRpcServer(object):
    def __init__(self,rpc_queue):

        connection = pika.BlockingConnection(pika.ConnectionParameters(
                                    host='localhost'))

        channel = connection.channel()

        channel.queue_declare(queue='rpc_queue')

        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(self.on_request, queue='rpc_queue')

        print " Serving requests"
        channel.start_consuming()

    def on_request(self,ch,method,props,body):
        n = int(body)

        print " computing square(%s)" % (n,)
        response = square(n)

        ch.basic_publish(exchange='',
                         routing_key=props.reply_to,
                         properties=pika.BasicProperties(
                            correlation_id=props.correlation_id),
                         body=str(response))

        ch.basic_ack(delivery_tag = method.delivery_tag)

server = NeonRpcServer('rpc_queue')
