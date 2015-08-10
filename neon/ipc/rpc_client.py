from rpc import RpcClient
import sys

if len(sys.argv) != 2:
    print "Usage: python rpc_client <rpc_queue_name>"
    sys.exit(1)

# declare an rpc client listening on queue specified by first arg
neon_rpc = RpcClient(sys.argv[1])

arg = int(raw_input("Give an integer to pow: "))

print " [x] Making request"
response = neon_rpc.call(arg)
print " [.] Got %r" % (response,)
