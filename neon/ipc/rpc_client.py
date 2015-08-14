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
