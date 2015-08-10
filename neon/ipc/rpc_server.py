from rpc import RpcServer

def square(n):
    return n * n

server = RpcServer('my_rpc',square)
