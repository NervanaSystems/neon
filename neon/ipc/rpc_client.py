from rpc import RpcClient
import sys

if len(sys.argv) != 2:
    print "Please supply rpc queue name"
    sys.exit(1)

neon_rpc = RpcClient(sys.argv[1])

arg = int(raw_input("Give an integer to square: "))

print " [x] Making request"
response = neon_rpc.call(arg)
print " [.] Got %r" % (response,)
