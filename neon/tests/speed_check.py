# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
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
"""
Speed check
"""
import argparse
import logging
import os
import sys
import time

from neon.backends import gen_backend
from neon.util.persist import deserialize


def parse_args():
    parser = argparse.ArgumentParser(description='Run speed check examples')
    parser.add_argument('--cpu', default=0, help='Run CPU speed check',
                        type=int)
    parser.add_argument('--gpu', default="", help='Run GPU speed check '
                        '(specify one of cudanet or nervanagpu)')
    return parser.parse_args()


def speed_check(conf_file, num_epochs, **be_args):
    experiment = deserialize(os.path.join(dir, conf_file))
    experiment.model.num_epochs = num_epochs
    backend = gen_backend(model=experiment.model, **be_args)
    experiment.initialize(backend)
    start = time.time()
    experiment.run()
    return (time.time() - start)


if __name__ == '__main__':
    tot_time = 0
    args = parse_args()
    # setup an initial console logger (may be overridden in config)
    logging.basicConfig(level=40)  # ERROR or higher
    script_dir = os.path.dirname(os.path.realpath(__file__))
    check_file = os.path.join(script_dir, '..', '..', 'examples',
                              'convnet', 'synthetic-sanity_check.yaml')
    # NOTE: number of epochs calibrated to take ~30 seconds total on 2014 MBP
    for be, num_epochs in [("cpu", 120), ("gpu", 225)]:
        be_args = {'rng_seed': 0}
        if (args.__dict__[be] != 0 and args.__dict__[be] != "" and
                args.__dict__[be] != "0"):
            if be != "cpu":
                be_args[be] = 1
            if be == "gpu":
                be_args[be] = args.__dict__[be]
            print('{} time: '.format(be)),
            sys.stdout.flush()
            res_time = speed_check(check_file, num_epochs, **be_args)
            print('{:.1f}s'.format(res_time))
            tot_time += res_time
    if tot_time > 0:
        print('total time: {:.1f}s'.format(tot_time))
    sys.exit(0)
