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

import numpy as np
import os

from neon.data import DataIterator, load_mnist
from neon.initializers import Gaussian, Constant
from neon.layers import GeneralizedCost, Affine, BatchNorm, Dropout, Conv, Pooling, MergeConcat
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary
from neon.util.persist import save_obj


def test_model_serialize(backend):
    (X_train, y_train), (X_test, y_test), nclass = load_mnist()
    train_set = DataIterator([X_train, X_train], y_train, nclass=nclass, lshape=(1, 28, 28))

    init_norm = Gaussian(loc=0.0, scale=0.01)

    # initialize model
    path1 = [Conv((5, 5, 16), init=init_norm, bias=Constant(0), activation=Rectlin()),
             Pooling(2),
             Affine(nout=20, init=init_norm, bias=init_norm, activation=Rectlin())]
    path2 = [Dropout(keep=0.5),
             Affine(nout=20, init=init_norm, bias=init_norm, activation=Rectlin())]
    layers = [MergeConcat([path1, path2]),
              Affine(nout=20, init=init_norm, bias=init_norm, activation=Rectlin()),
              BatchNorm(),
              Affine(nout=10, init=init_norm, activation=Logistic(shortcut=True))]

    tmp_save = 'test_model_serialize_tmp_save.pickle'
    mlp = Model(layers=layers)
    mlp.optimizer = GradientDescentMomentum(learning_rate=0.1, momentum_coef=0.9)
    mlp.cost = GeneralizedCost(costfunc=CrossEntropyBinary())

    n_test = 3
    num_epochs = 3
    # Train model for num_epochs and n_test batches
    for epoch in range(num_epochs):
        for i, (x, t) in enumerate(train_set):
            x = mlp.fprop(x)
            delta = mlp.cost.get_errors(x, t)
            mlp.bprop(delta)
            mlp.optimizer.optimize(mlp.layers_to_optimize, epoch=epoch)
            if i > n_test:
                break

    # Get expected outputs of n_test batches and states of all layers
    outputs_exp = []
    pdicts_exp = [l.get_params_serialize() for l in mlp.layers_to_optimize]
    for i, (x, t) in enumerate(train_set):
        outputs_exp.append(mlp.fprop(x, inference=True))
        if i > n_test:
            break

    # Serialize model
    save_obj(mlp.serialize(keep_states=True), tmp_save)

    # Load model
    mlp = Model(layers=layers)
    mlp.load_weights(tmp_save)

    outputs = []
    pdicts = [l.get_params_serialize() for l in mlp.layers_to_optimize]
    for i, (x, t) in enumerate(train_set):
        outputs.append(mlp.fprop(x, inference=True))
        if i > n_test:
            break

    # Check outputs, states, and params are the same
    for output, output_exp in zip(outputs, outputs_exp):
        assert np.allclose(output.get(), output_exp.get())

    for pd, pd_exp in zip(pdicts, pdicts_exp):
        for s, s_e in zip(pd['states'], pd_exp['states']):
            if isinstance(s, list):  # this is the batch norm case
                for _s, _s_e in zip(s, s_e):
                    assert np.allclose(_s, _s_e)
            else:
                assert np.allclose(s, s_e)
        for p, p_e in zip(pd['params'], pd_exp['params']):
            if isinstance(p, list):  # this is the batch norm case
                for _p, _p_e in zip(p, p_e):
                    assert np.allclose(_p, _p_e)
            else:
                assert np.allclose(p, p_e)

    os.remove(tmp_save)
