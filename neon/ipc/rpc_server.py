from rpc import RpcServer
import sys

if len(sys.argv) != 2:
    print "Please supply rpc queue name"
    sys.exit(1)

def square(n):
    return n * n

server = RpcServer(sys.argv[1],square)
