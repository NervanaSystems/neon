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

"""
Example that creates and uses a network without a configuration file.
"""

import logging
from neon.backends import gen_backend
from neon.layers import FCLayer, DataLayer, CostLayer
from neon.models import MLP
from neon.transforms import RectLin, Logistic, CrossEntropy
from neon.datasets import MNIST
from neon.experiments import FitPredictErrorExperiment

logging.basicConfig(level=20)
logger = logging.getLogger()


def create_model(nin):
    layers = []
    layers.append(DataLayer(nout=nin))
    layers.append(FCLayer(nout=100, activation=RectLin()))
    layers.append(FCLayer(nout=10, activation=Logistic()))
    layers.append(CostLayer(cost=CrossEntropy()))
    model = MLP(num_epochs=10, batch_size=128, layers=layers)
    return model


def run():
    model = create_model(nin=784)
    backend = gen_backend(rng_seed=0)
    dataset = MNIST(repo_path='~/data/')
    experiment = FitPredictErrorExperiment(model=model,
                                           backend=backend,
                                           dataset=dataset)
    experiment.run()


if __name__ == '__main__':
    run()
