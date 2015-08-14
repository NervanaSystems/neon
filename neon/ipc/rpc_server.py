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
from rpc import RpcServer
import sys

if len(sys.argv) != 3:
    print "Usage: python rpc_server <rpc queue name> <power>"
    sys.exit(1)


def func(k):
    return lambda x: pow(int(x), k)

k = int(sys.argv[2])

# start an rpc server listening on queue specified by first argument
# and applying procdure that takes integer arguments to the k-th power
server = RpcServer(sys.argv[1], func(k))
