from neon.ipc.shmem import Server
import logging

logging.basicConfig(level=20)
logger = logging.getLogger(__name__)

#server1 = Server(req_size=1, res_size=1, channel_id="one")

#server2 = Server(req_size=1, res_size=1, channel_id="two")

# Generate posix ipc components with default name
server3 = Server(req_size=1, res_size=1) 

server3.send('x')

while True:
    data = server3.receive()
    print "received: ", data
    server3.send(data)
     
