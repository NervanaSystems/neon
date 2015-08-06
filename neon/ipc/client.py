from neon.ipc.shmem import Client
import logging

logging.basicConfig(level=20)
logger = logging.getLogger(__name__)

#client1 = Client(channel_id="one")

#client2 = Client(channel_id="two")

# Open a client connection to a server with the default name
client = Client()

# The below will fail if we have not started a server
# with channel_id "three"
#client3 = Client(channel_id="three")

data = client.receive()

while True:
    string = raw_input('---> ')
    client.send(string) 
    data = client.receive()
    print "received: ", data
