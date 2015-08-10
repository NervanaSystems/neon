from rpc import RpcServer
import sys

if len(sys.argv) != 3:
    print "Usage: python rpc_server <rpc queue name> <power>"
    sys.exit(1)

def square(n):
    return n * n

def func(k):
    return lambda x: pow(x,k)

k = int(sys.argv[2])
server = RpcServer(sys.argv[1],func(k))
