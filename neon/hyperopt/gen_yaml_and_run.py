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
hyperopt script: spearmint calls into this file's main() function with the
current set of hyperparameters (selected by spearmint). It then:
- reads the hyper-yaml file
- parses the parameters suggested by spearmint
- generates a temp yaml file
- runs neon
- gets the outputs
"""

import os
import time
import logging
from neon.backends import gen_backend
from neon.util.persist import deserialize
from neon.metrics.misclass import MisclassPercentage


def main(job_id, params):
    print('spear_wrapper job #:%s' % str(job_id))
    print("spear_wrapper in directory: %s" % os.getcwd())
    print("spear_wrapper params are:%s" % params)

    return call_neon(params)


def call_neon(params):
    """
    runs the system call to neon and reads the result to give back to sm
    """
    timestring = str(int(time.time()))
    experiment_dir = os.path.realpath(os.environ['HYPEROPT_PATH'])
    # Generate the yaml file
    hyper_file = os.path.join(experiment_dir, 'hyperyaml.yaml')
    yaml_file = os.path.join(experiment_dir, 'yamels',
                             'temp' + timestring + '.yaml')
    try:
        os.mkdir('yamels')
    except OSError:
        "Directory exists"
    write_params(hyper_file, yaml_file, params)

    # Initialize the neon experiment
    logging.basicConfig(level=20)
    experiment = deserialize(yaml_file)
    backend = gen_backend(model=experiment.model)  # , gpu='nervanagpu'
    experiment.initialize(backend)

    # ensure TOP1 error is calculated
    if not hasattr(experiment, 'metrics'):
        experiment.metrics = {'validation': [MisclassPercentage(error_rank=1)],
                              'test': [MisclassPercentage(error_rank=1)]}
    for item in ['validation', 'test']:
        if item not in experiment.metrics:
            experiment.metrics[item] = [MisclassPercentage(error_rank=1)]
        metriclist = [str(x) for x in experiment.metrics[item]]
        if 'MisclassPercentage_TOP_1' not in metriclist:
            experiment.metrics[item].append(MisclassPercentage(error_rank=1))

    result = experiment.run()

    # check if validation set is available
    if experiment.dataset.has_set('validation'):
        hyperopt_set = 'validation'
    elif experiment.dataset.has_set('test'):
        hyperopt_set = 'test'
        print("Warning: No validation set found, performing hyperparameter "
              "optimization on test set.")
    else:
        raise AttributeError("No error found.")

    return result[hyperopt_set]['MisclassPercentage_TOP_1']


def write_params(input_file, output_file, params):
    """
    go thorugh the hyperyaml line by line to create tempyaml
    """
    with open(input_file, 'r') as fin:
        with open(output_file, 'w') as fout:
            for line in fin:
                if '!hyperopt' in line:
                    line = parse_line(line, params)
                fout.write(line)


def parse_line(line, params):
    """
    Replace the line defining the parameter range by just a name value pair.
    """
    dic = [k.strip("{},") for k in line.split()]
    out = params[dic[2]][0]
    return dic[0] + " " + str(out) + ",\n"
