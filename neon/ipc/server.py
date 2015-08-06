from neon.ipc.shmem import Server
import logging

logging.basicConfig(level=20)
logger = logging.getLogger(__name__)

print "server"
server1 = Server(req_size=1, res_size=1, channel_id="one")
server1.start()

#server2 = Server(req_size=1, res_size=1, channel_id="two")
#server2.start()

while True:
    c = 8
