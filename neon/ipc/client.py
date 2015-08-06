from neon.ipc.shmem import Client
import logging

logging.basicConfig(level=30)
logger = logging.getLogger(__name__)

print "client"
client1 = Client()
client1.start()

client2 = Client()
client2.start()

while True:
    c = 8
