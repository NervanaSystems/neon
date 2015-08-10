from rpc import RpcServer
import sys

if len(sys.argv) != 3:
    print "Usage: python rpc_server <rpc queue name> <power>"
    sys.exit(1)

def func(k):
    return lambda x: pow(int(x),k)

k = int(sys.argv[2])

# start an rpc server listening on queue specified by first argument
# and applying procdure that takes integer arguments to the k-th power
server = RpcServer(sys.argv[1],func(k))
