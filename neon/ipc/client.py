from neon.ipc.shmem import Client
import logging

logging.basicConfig(level=30)
logger = logging.getLogger(__name__)

print "client"
client1 = Client(channel_id="one")

client2 = Client(channel_id="two")

# The below will fail
#client3 = Client(channel_id="three")
while True:
    pass
