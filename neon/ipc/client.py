# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import logging

from neon.ipc.shmem import Client

logging.basicConfig(level=20)
logger = logging.getLogger(__name__)

# client1 = Client(channel_id="one")

# client2 = Client(channel_id="two")

# Open a client connection to a server with the default name
client = Client()

# The below will fail if we have not started a server
# with channel_id "three"
# client3 = Client(channel_id="three")

data = client.receive()

while True:
    string = raw_input('---> ')
    client.send(string)
    data = client.receive()
    print "received: ", data
