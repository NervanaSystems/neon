# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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

from neon.backends import gen_backend
from neon.initializers.initializer import Array
from neon.layers.recurrent import LSTM
from neon.layers.container import Seq2Seq
from neon.transforms import Tanh, Logistic

import numpy as np
from neon import NervanaObject
from neon.util.beamsearch import BeamSearch
from utils import allclose_with_out


def reformat_samples(seq2seq_obj, num_beams, batch_size):
    samples = [[seq2seq_obj.candidates[bb][:, ex]
                for bb in range(num_beams)]
               for ex in range(batch_size)]
    examples = []
    for ex in range(batch_size):
        examples.append(np.vstack([samples[ex][ii] for ii in range(num_beams)]))
    return examples


def test_beamsearch(backend_default):
    """
    Simlulated beam search on a minibatch of 2, for 4 time steps. The
    LSTM states are real but the "softmax outputs" z are hardcoded and
    not taken from  the network.
    There are 6 tokens the network outputs, and they have probabilities
    like exp(1), exp(5), exp(7)

    The test asserts that the score_lists assigned by _beamsearch_step(z_list)
    are equal to the probabilities computed manually adding probabilities
    to z_list.
    """
    be = backend_default

    batch_size = 2
    be.bsz = batch_size
    time_steps = 4
    nout = 6
    num_beams = 3

    # create unused layers
    activation = Tanh()
    gate_activation = Logistic()
    init_ary = np.eye(nout)
    init = Array(init_ary)
    encoder = LSTM(nout, init,
                   activation=activation, gate_activation=gate_activation,
                   name="Enc")
    decoder = LSTM(nout, init,
                   activation=activation, gate_activation=gate_activation,
                   name="Dec")

    class DummyFProp():
        """
        Constructs an artificial beam search example with known correct outputs.
        This is called inside a nested loop over steps, num_life. In the first
        time step there is one life beam, after that, 3 life beams per step.
        There are 4 time steps total. Each beamsearch_step builds one list over
        num_life beams.

        At t=0, the winners for ex0 are 1, 4, 5 (indexed by their position) and
        winners for ex1 are 2,4,5. From there we continue the beam for ex0:
            12, 13, 14              6+2=8 6+3=9  6+2=8
            40, 43, 45  with scores 5+4=9 5+3=8  5+7=12 three new winners 45, 52, 55
            50, 52, 55              5+4=9 5+6=11 5+5=10

        for ex2
            1 4 5  with scores   5 4 7
        we get the three winners 1, 4, 5 and continue (just taking the
        3 in order, no sorting)
            10 12 13 14 (not unique!)  5+2=7  5+2=7  5+3=8
            41 42 43       with scores 4+6=10 4+5=9  4+7=11 winners  43 51 52
            51 52 53                   7+4=11 7+6=13 7+3=10 scores   11 11 13
        continue from the three winners 43 51 52
            431 433 434             11+10=21 11+3=14 11+9=20
            511 512 513 with scores 11+6=17  11+5=16 11+7=18  winners 431 434 520
            520 521 522             13+8=21  13+4=17 13+6=19  scores   21  20  21
        continue from three winners 431 511 513 (going along beams, the matches
        in a beam)
            4310 4312 4313 4314             21+2=23  21+2=23 21+3=24 21+10=31 (not unique!)
            4341 4342 4343      with scores 20+10=30 20+5=25 20+7=27        winners 4314 4341 5204
            5200 5202 5204                  21+8=29  21+6=27 21+10=31       scores    31   30   31
        overall winners are 4314 4341 5204

        """
        def __init__(self):
            self.i = -1
            # t=0
            #                                 X        x  x  <-- winners: 1, 4, 5  (for example 0)
            z = be.array(np.exp(np.array([[1, 6, 2, 1, 5, 5],
                                          [1, 5, 2, 2, 4, 7]]))).T

            # t=1
            #                                     x  x  x  <-- give we picked 4: new winners 2,3,4
            z1 = be.array(np.exp(np.array([[1, 1, 2, 3, 2, 1],
                                           [2, 1, 2, 3, 2, 1]]))).T
            #                               x        x     x  <-- give we picked 5:
            #                                                     new winners 0,3,[5]
            #                                                     score 12
            z2 = be.array(np.exp(np.array([[4, 1, 2, 3, 1, 7],
                                           [2, 6, 5, 7, 2, 4]]))).T
            #                               x     X        X  <-- give we picked 1:
            #                                                     new winners 0,[2],[5]
            #                                                     scores 12, 11
            z3 = be.array(np.exp(np.array([[4, 1, 6, 3, 1, 5],
                                           [1, 4, 6, 3, 2, 1]]))).T

            # t=2
            # example 0: given constructed (1, 5), score 11: 3, 4; scores 21, 20
            z4 = be.array(np.exp(np.array([[1, 1, 2, 10, 9, 1],
                                           [2, 10, 2, 3, 9, 1]]))).T
            # example 0: given constructed (5, 5), score 12: none selected from this beam
            z5 = be.array(np.exp(np.array([[4, 1, 2, 3, 1, 7],
                                           [2, 6, 5, 7, 2, 4]]))).T
            # example 0: given constructed (1, 2), score 12: 1; score 20
            z6 = be.array(np.exp(np.array([[4, 8, 6, 3, 1, 5],
                                           [8, 4, 6, 3, 1, 1]]))).T

            # t=3
            # example 0: given constructed (1, 5, 4), score 20: 1, score 30
            z7 = be.array(np.exp(np.array([[1, 10, 2, 1, 1, 1],
                                           [2, 1, 2, 3, 10, 1]]))).T
            # example 0: given constructed (1, 2, 1), score 20: 5, score 30
            z8 = be.array(np.exp(np.array([[4, 1, 2, 3, 1, 10],
                                           [2, 10, 5, 7, 2, 4]]))).T
            # example 0: given constructed (1, 5, 3), score 21: 4, score 31
            z9 = be.array(np.exp(np.array([[4, 8, 6, 3, 10, 5],
                                           [8, 4, 6, 3, 10, 1]]))).T

            self.z_list = [z, z1, z2, z3, z4, z5, z6, z7, z8, z9]

        def fprop(self, z, inference=True, init_state=None):
            self.i += 1
            return self.z_list[self.i]

    def final_state():
        return be.zeros_like(decoder.h[-1])

    class InObj(NervanaObject):
        def __init__(self):
            self.shape = (nout, time_steps)
            self.decoder_shape = (nout, time_steps)

    decoder.fprop = DummyFProp().fprop
    layers = Seq2Seq([encoder, decoder], decoder_connections=[0])
    layers.decoder._recurrent[0].final_state = final_state

    in_obj = InObj()
    layers.configure(in_obj)  # made zeros because zeros have shape
    layers.allocate()
    layers.allocate_deltas(None)
    beamsearch = BeamSearch(layers)
    inputs = be.iobuf(in_obj.shape)
    beamsearch.beamsearch(inputs, num_beams=num_beams)

    ex0 = np.array([[1, 5, 4, 1],
                    [1, 2, 1, 5],
                    [1, 5, 3, 4]])
    ex1 = np.array([[5, 1, 4, 4],
                    [5, 1, 1, 1],
                    [5, 2, 0, 4]])

    # extract all candidates
    examples = reformat_samples(beamsearch, num_beams, batch_size)
    assert allclose_with_out(examples[0], ex0)
    assert allclose_with_out(examples[1], ex1)

if __name__ == '__main__':
    be = gen_backend(backend='gpu', batch_size=2)
    test_beamsearch(be)
