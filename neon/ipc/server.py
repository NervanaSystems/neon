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

from neon.ipc.shmem import Server

logging.basicConfig(level=20)
logger = logging.getLogger(__name__)

# server1 = Server(req_size=1, res_size=1, channel_id="one")

# server2 = Server(req_size=1, res_size=1, channel_id="two")

# Generate posix ipc components with default name
server3 = Server(req_size=1, res_size=1)

server3.send('x')

while True:
    data = server3.receive()
    print "received: ", data
    server3.send(data)
