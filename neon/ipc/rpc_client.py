from rpc import RpcClient

neon_rpc = RpcClient('rpc_queue')

arg = int(raw_input("Give an integer to square: "))

print " [x] Making request"
response = neon_rpc.call(arg)
print " [.] Got %r" % (response,)
