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
Simple recurrent neural network with one hidden layer.
"""

import logging

from neon.backends.backend import Block
from neon.diagnostics.visualize_rnn import VisualizeRNN
from neon.models.mlp import MLP
from neon.util.compat import range
from neon.util.param import req_param

logger = logging.getLogger(__name__)


class RNN(MLP):
    """
    Recurrent neural network. Supports LSTM and standard RNN layers.
    """
    def __init__(self, **kwargs):
        self.accumulate = True
        # Reusing deltas not supported for RNNs yet
        self.reuse_deltas = False
        super(RNN, self).__init__(**kwargs)
        req_param(self, ['unrolls'])
        self.rec_layer = self.layers[1]

    def link(self, initlayer=None):
        """
        link function for the RNN differs from the MLP in that it does not
        print the layers
        """
        for ll, pl in zip(self.layers, [initlayer] + self.layers[:-1]):
            ll.set_previous_layer(pl)
        # self.print_layers()

    def fit(self, dataset):
        viz = VisualizeRNN()
        error = self.backend.empty((1, 1))
        mb_id = self.backend.empty((1, 1))
        self.print_layers()
        self.data_layer.init_dataset(dataset)
        self.data_layer.use_set('train')
        # "output":"input":"rec"
        #           "lstm_x":"lstm_ih":"lstm_fh":"lstm_oh":"lstm_ch"
        self.grad_checker(numgrad="output")
        logger.info('commencing model fitting')
        errorlist = []
        suberrorlist = []
        suberror = self.backend.zeros((1, 1))
        while self.epochs_complete < self.num_epochs:
            self.backend.begin(Block.epoch, self.epochs_complete)
            error.fill(0.0)
            mb_id = 1
            self.data_layer.reset_counter()
            while self.data_layer.has_more_data():
                self.backend.begin(Block.minibatch, mb_id)
                self.reset(mb_id)
                self.backend.begin(Block.fprop, mb_id)
                self.fprop(debug=(True if (mb_id is -1) else False))
                self.backend.end(Block.fprop, mb_id)
                self.backend.begin(Block.bprop, mb_id)
                self.bprop(debug=(True if (mb_id is -1) else False))
                self.backend.end(Block.bprop, mb_id)
                self.backend.begin(Block.update, mb_id)
                self.update(self.epochs_complete)
                self.backend.end(Block.update, mb_id)

                self.cost_layer.cost.set_outputbuf(
                    self.class_layer.output_list[-1])
                suberror = self.cost_layer.get_cost()
                suberrorlist.append(float(suberror.asnumpyarray()))
                self.backend.add(error, suberror, error)
                if self.step_print > 0 and mb_id % self.step_print == 0:
                    logger.info('%d.%d logloss=%0.5f', self.epochs_complete,
                                mb_id / self.step_print - 1,
                                float(error.asnumpyarray()) /
                                self.data_layer.num_batches)
                self.backend.end(Block.minibatch, mb_id)
                mb_id += 1
            self.backend.end(Block.epoch, self.epochs_complete)
            self.epochs_complete += 1
            errorlist.append(float(error.asnumpyarray()) /
                             self.data_layer.num_batches)
            # self.print_layers(debug=True)
            logger.info('epoch: %d, total training error: %0.5f',
                        self.epochs_complete, float(error.asnumpyarray()) /
                        self.data_layer.num_batches)
            if self.make_plots is True:
                self.plot_layers(viz, suberrorlist, errorlist)

        self.data_layer.cleanup()

    def reset(self, batch):
        """
        instead of having a separate buffer for hidden_init, we are now
        using the last element output_list[-1] for that.
        The shuffle is no longer necessary because fprop directly looks
        into the output_list buffer.
        """
        if (batch % self.reset_period) == 0 or batch == 1:
            self.rec_layer.output_list[-1].fill(0)  # reset fprop state
            self.rec_layer.deltas.fill(0)  # reset bprop (for non-truncated)
            if 'c_t' in self.rec_layer.__dict__:
                self.rec_layer.c_t[-1].fill(0)
                self.rec_layer.celtas.fill(0)

    def plot_layers(self, viz, suberrorlist, errorlist):
        # generic error plot
        viz.plot_error(suberrorlist, errorlist)

        # LSTM specific plots
        if 'c_t' in self.rec_layer.__dict__:
            viz.plot_lstm_wts(self.rec_layer, scale=1.1, fig=4)
            viz.plot_lstm_acts(self.rec_layer, scale=21, fig=5)
        # RNN specific plots
        else:
            viz.plot_weights(self.rec_layer.weights.asnumpyarray(),
                             self.rec_layer.weights_rec.asnumpyarray(),
                             self.class_layer.weights.asnumpyarray())
            viz.plot_activations(self.rec_layer.pre_act_list,
                                 self.rec_layer.output_list,
                                 self.class_layer.pre_act_list,
                                 self.class_layer.output_list,
                                 self.cost_layer.targets)

    def fprop(self, debug=False, eps_tau=-1, eps=0,
              num_target=None, num_i=0, num_j=0):
        """
        Adding numerical gradient functionality here to avoid duplicate fprops.
        TODO: Make a version where the for tau loop is inside the layer. The
        best way is to have a baseclass for both RNN and LSTM for this.
        """
        self.data_layer.fprop(None)  # get next mini batch
        inputs = self.data_layer.output
        y = self.rec_layer.output_list  # note: just a shorthand, no copy.
        c = [None for k in range(len(y))]
        if 'c_t' in self.rec_layer.__dict__:
            c = self.rec_layer.c_t

        # loop for rec_layer
        for tau in range(0, self.unrolls):
            if tau == eps_tau:
                numpy_target = num_target[num_i, num_j].asnumpyarray()
                num_target[num_i, num_j] = (numpy_target + eps)
            if debug:
                logger.debug("in RNNB.fprop, tau %d, input %d" % (tau,
                             inputs[tau].asnumpyarray().argmax(0)[0]))
            self.rec_layer.fprop(y[tau-1], c[tau-1], inputs[tau], tau)
            if tau == eps_tau:
                num_target[num_i, num_j] = numpy_target

        # loop for class_layer
        for tau in range(0, self.unrolls):
            if tau == eps_tau:
                numpy_target = num_target[num_i, num_j].asnumpyarray()
                num_target[num_i, num_j] = (numpy_target + eps)
            if debug:
                logger.debug("in RNNB.fprop, tau %d, input %d" % (tau,
                             inputs[tau].asnumpyarray().argmax(0)[0]))
            self.class_layer.fprop(y[tau], tau)
            if tau == eps_tau:
                num_target[num_i, num_j] = numpy_target
        # cost layer fprop is a pass.

    def bprop(self, debug, numgrad=None):
        """
        Parent method for bptt and truncated-bptt. Truncation is neccessary
        for the standard RNN as a way to prevent exploding gradients. For the
        LSTM it also
        """
        if self.truncate:
            self.trunc_bprop_tt(debug, numgrad)
        else:
            self.bprop_tt(debug, numgrad)

    def trunc_bprop_tt(self, debug, numgrad=None):
        """
        TODO: move the loop over t into the layer class.
        """
        if numgrad is None:
            min_unroll = 1
        else:
            logger.debug("MLP.bprop single unrolling for numgrad")
            min_unroll = self.unrolls

        for tau in range(min_unroll-0, self.unrolls+1):
            self.cost_layer.cost.set_outputbuf(
                self.class_layer.output_list[tau-1])
            self.cost_layer.bprop(None, tau-1)
            if debug:
                tmp = self.cost_layer.targets[tau-1].asnumpyarray()
                tmp = tmp.argmax(0)[0]
                logger.debug("in RNNB.bprop, tau %d target %d" % (tau-1, tmp))
            error = self.cost_layer.deltas
            self.class_layer.bprop(error, tau, numgrad=numgrad)
            error = self.class_layer.deltas
            for t in list(range(0, tau))[::-1]:
                if 'c_t' in self.rec_layer.__dict__:
                    cerror = self.rec_layer.celtas  # on t=0, prev batch state
                else:
                    cerror = None  # for normal RNN
                self.rec_layer.bprop(error, cerror, t, numgrad=numgrad)
                error[:] = self.rec_layer.deltas  # [TODO] why need deepcopy?

    def bprop_tt(self, debug, numgrad=None):
        """
        Keep state over consecutive unrollings. Explodes for RNN, and is not
        currently used for anything, but future recurrent layers might use it.
        """

        temp1 = self.backend.zeros(self.class_layer.deltas.shape)
        temp2 = self.backend.zeros(self.class_layer.deltas.shape)
        temp1c = self.backend.zeros(self.class_layer.deltas.shape)
        temp2c = self.backend.zeros(self.class_layer.deltas.shape)

        for tau in list(range(self.unrolls))[::-1]:
            self.cost_layer.cost.set_outputbuf(
                self.class_layer.output_list[tau])
            self.cost_layer.bprop(None, tau)
            cost_error = self.cost_layer.deltas
            self.class_layer.bprop(cost_error, tau, numgrad=numgrad)

            external_error = self.class_layer.deltas
            internal_error = self.rec_layer.deltas
            if 'c_t' in self.rec_layer.__dict__:
                internal_cerror = self.rec_layer.celtas
                external_cerror = self.backend.zeros(temp1.shape)
            else:
                internal_cerror = None
                external_cerror = None

            self.rec_layer.bprop(external_error, external_cerror, tau,
                                 numgrad=numgrad)
            temp1[:] = self.rec_layer.deltas
            if 'c_t' in self.rec_layer.__dict__:
                temp1c[:] = self.rec_layer.celtas
            self.rec_layer.bprop(internal_error, internal_cerror, tau,
                                 numgrad=numgrad)
            temp2[:] = self.rec_layer.deltas
            if 'c_t' in self.rec_layer.__dict__:
                temp2c[:] = self.rec_layer.celtas
            self.backend.add(temp1, temp2, out=self.rec_layer.deltas)
            if 'c_t' in self.rec_layer.__dict__:
                self.backend.add(temp1c, temp2c, out=self.rec_layer.celtas)

    def grad_checker(self, numgrad="lstm_ch"):
        """
        Check gradients for LSTM layer:
          - W is replicated, only inject the eps once, repeat, average.
            bProp is only through the full stack, but wrt. the W in each
            level. bProp does this through a for t in tau.

            Need a special fprop that injects into one unrolling only.
        """
        for layer in self.layers:
            logger.info("%s", str(layer))

        if numgrad is "output":
            num_target = self.class_layer.weights
            anl_target = self.class_layer.weight_updates
            num_i, num_j = 15, 56
        elif numgrad is "input":
            num_target = self.rec_layer.weights
            anl_target = self.rec_layer.weight_updates
            num_i, num_j = 12, 110  # 110 is "n"
        elif numgrad is "rec":
            num_target = self.rec_layer.weights_rec
            anl_target = self.rec_layer.updates_rec
            num_i, num_j = 12, 63
        elif numgrad is "lstm_x":
            num_target = self.rec_layer.Wfx
            anl_target = self.rec_layer.Wfx_updates
            num_i, num_j = 12, 110
        elif numgrad is "lstm_ih":
            num_target = self.rec_layer.Wih
            anl_target = self.rec_layer.Wih_updates
            num_i, num_j = 12, 55
        elif numgrad is "lstm_fh":
            num_target = self.rec_layer.Wfh
            anl_target = self.rec_layer.Wfh_updates
            num_i, num_j = 12, 55
        elif numgrad is "lstm_oh":
            num_target = self.rec_layer.Woh
            anl_target = self.rec_layer.Woh_updates
            num_i, num_j = 12, 55
        elif numgrad is "lstm_ch":
            num_target = self.rec_layer.Wch
            anl_target = self.rec_layer.Wch_updates
            num_i, num_j = 12, 55

        eps = 1e-6  # better to use float64 in cpu.py for this
        numerical = 0  # initialize buffer
        #  loop to inject epsilon in different unrolling stages
        for eps_tau in range(0, self.unrolls):
            self.reset(1)  # clear hidden input
            self.fprop(debug=False, eps_tau=eps_tau, eps=0,
                       num_target=num_target, num_i=num_i, num_j=num_j)
            self.cost_layer.set_targets()
            self.data_layer.reset_counter()
            self.cost_layer.cost.set_outputbuf(
                self.class_layer.output_list[-1])
            suberror_eps = self.cost_layer.get_cost().asnumpyarray()

            self.reset(1)
            self.fprop(debug=False, eps_tau=eps_tau, eps=eps,
                       num_target=num_target, num_i=num_i, num_j=num_j)
            self.data_layer.reset_counter()
            self.cost_layer.cost.set_outputbuf(
                self.class_layer.output_list[-1])
            suberror_ref = self.cost_layer.get_cost().asnumpyarray()

            num_part = (suberror_eps - suberror_ref) / eps
            logger.debug("numpart for  eps_tau=%d of %d is %e",
                         eps_tau, self.unrolls, num_part)
            numerical += num_part

        # bprop for analytical gradient
        self.bprop(debug=False, numgrad=numgrad)

        analytical = anl_target[num_i, num_j].asnumpyarray()
        logger.debug("---------------------------------------------")
        logger.debug("RNN grad_checker: suberror_eps %f", suberror_eps)
        logger.debug("RNN grad_checker: suberror_ref %f", suberror_ref)
        logger.debug("RNN grad_checker: numerical %e", numerical)
        logger.debug("RNN grad_checker: analytical %e", analytical)
        logger.debug("RNN grad_checker: ratio %e", 1./(numerical/analytical))
        logger.debug("---------------------------------------------")

    def predict_generator(self, dataset, setname):
        """
        Generate flattened predicitons and true labels for the given dataset,
        one mini-batch at a time.

        Agruments:
            dataset: A neon dataset instance
            setname: Which set to compute predictions for (test, train, val)

        Returns:
            tuple: on each call will yield a 2-tuple of outputs and references.
                   The first item is the model probabilities for each class,
                   and the second item is either the one-hot or raw labels with
                   ground truth.

        See Also:
            predict_fullset
        """
        self.data_layer.init_dataset(dataset)
        assert self.data_layer.has_set(setname)
        self.data_layer.use_set(setname, predict=True)
        self.data_layer.reset_counter()

        predlabels = self.backend.empty((1, self.batch_size))
        labels = self.backend.empty((1, self.batch_size))

        # TODO: find some alternate way of re-assembling data that doesn't
        # require allocating space for the entire dataset.
        outputs_pred = self.backend.zeros((self.data_layer.num_batches *
                                           self.unrolls, self.batch_size))
        outputs_targ = self.backend.zeros((self.data_layer.num_batches *
                                           self.unrolls, self.batch_size))

        mb_id = 0
        self.data_layer.reset_counter()
        self.set_train_mode(False)
        while self.data_layer.has_more_data():
            mb_id += 1
            self.reset(mb_id)
            self.fprop(debug=False)
            # time unrolling loop to disseminate fprop results
            for tau in range(self.unrolls):
                probs = self.class_layer.output_list[tau]
                targets = self.data_layer.targets[tau]
                self.backend.argmax(targets, axis=0, out=labels)
                self.backend.argmax(probs, axis=0, out=predlabels)

                # collect batches to re-assemble continuous data
                idx = self.unrolls * (mb_id - 1) + tau
                outputs_pred[idx, :] = predlabels
                outputs_targ[idx, :] = labels

        self.data_layer.cleanup()

        # flatten the 2d predictions into our canonical 1D format
        pred_flat = outputs_pred.transpose().reshape((1, -1))
        targ_flat = outputs_targ.transpose().reshape((1, -1))

        for i in range(self.data_layer.num_batches):
            start = i * self.unrolls * self.batch_size
            end = start + (self.unrolls * self.batch_size)
            yield (pred_flat[start:end], targ_flat[start:end])

    def write_string(self, pred, targ, setname):
            """ For text prediction, reassemble the batches and print out a
            short contigous segment of target text and predicted text - useful
            to check for off-by-one errors and the like"""
            import numpy as np

            pred_int = pred[0, 2:40].asnumpyarray().ravel().astype(np.int8)
            targ_int = targ[0, 2:40].asnumpyarray().ravel().astype(np.int8)
            # remove special characters, replace them with '#'
            pred_int[pred_int < 32] = 35
            targ_int[targ_int < 32] = 35

            # create output strings
            logging.info("the target for '%s' is: '%s'", setname,
                         ''.join(targ_int.view('c')))
            logging.info("prediction for '%s' is: '%s'", setname,
                         ''.join(pred_int.view('c')))
