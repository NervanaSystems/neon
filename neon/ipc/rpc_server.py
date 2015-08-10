from rpc import RpcServer

def square(n):
    return n * n

server = RpcServer('rpc_queue',square)
