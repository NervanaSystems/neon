In order to use the rpc library, you need an AMQP server and you need pika.

To see the rpc library in action, execute the following in 2 different terminals:

1. `python rpc_server.py two 2`
2. `python rpc_server.py three 3`

The first call starts an rpc server listening on a message queue named "two"
that will square all integer arguments it receives. 

The second call starts an rpc server listening on a message queue named "three"
that will take all integer arguments it receives to the third power.

In a third terminal window, execute commands of the form:
   `python rpc_client.py <two|three>` 

You will be prompted for integer arguments, which will generate responsed from the 
rpc server listening on the specified message queue name.

Observe that if you don't specify one of "two" or "three" as the message queue name,
the messages are lost.
