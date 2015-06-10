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
from neon.util.param import req_param, opt_param, ensure_dtype

logger = logging.getLogger(__name__)


class RNN(MLP):
    """
    **Recurrent Neural Network**

    Neon supports standard Recurrent Neural Networks (RNNs) as well as RNNs
    with Long Short Term Memory cells (LSTMs). These models are trained on
    sequence data, and therefore require a separate dataset format. Neon is
    distributed with the Moby Dick dataset, which is a character-based encoding
    of the book Moby Dick. Each character is represented in a one-hot encoding
    as one of the 128 lowest ASCII chars.

    *Dataset format and unrolling*

    For the purpose of illustration, assume the entire source text is the 30
    characters of ``'Your_shoe_fits_only_your_foot.'``. Note spaces have been
    replaced by underscores for readability. To create minibatches of size 2,
    we split the data into two subsequences ``'Your_shoe_fits_'`` and
    ``'only_your_foot.'`` which are treated as separate, independent sequences.

    The RNN is trained using truncated back-propagation through time (tBPTT),
    which means that the network in unrolled for a on number of steps,
    effectively turning it into a deep feed-forward network. To illustrate the
    process, consider an unrolling depth of 5 on the text above: The first step
    is to break each sequence into short chunks of the unrolling depth:

    | ``'Your_'  'shoe_'  'fits_'``
    | ``'only_'  'your_'  'foot.'``

    The second step is to create minibatches from the columns of this
    structure, e.g. the two sequences ``'Your_'`` and ``'only_'`` will form the
    first minibatch. This procedure leaves us with 3 minibatches in total.
    The reason for using columns rather than rows is that this way we start
    processing the independent sequences in parallel. Then, as we move to the
    next minibatch, we also move to the next consecutive time step, and
    immediately use the hidden state of the network that was computed at the
    previous time step.

    In the actual neon data format, each letter becomes a one-hot encoded
    vector, and thus each chunk is split up into a list over the unrolling
    steps, i.e.

    | ``'Your_'``
    | ``'only_'``

    becomes a list of tensors corresponding to the one-hot encodings of
    ``['Y', 'o'], ['o', 'n'], ['u', 'l'], ['r', 'y'], [' ', ' ']``.
    These lists form the elements of another list over the 3 minibatches that
    make up the full dataset.

    Note that in the more general case of datasets with multiple sequences of
    unequal lengths, it would be necessary to pick the product of unrolling
    steps and number of minibatches to be
    equal to the number of sequences, and the number of minibatches to be the
    length of the sequences. Sequences would need to be padded to the maximum
    length with an "empty character" code, e.g. the all-zeros vector rather
    than a one-hot encoding.

    In the Moby Dick example, the network is trained to predict one character
    ahead, so the targets used for training are simply a copy of the inputs
    shifted by one character into the future.

    """
    def __init__(self, **kwargs):
        self.accumulate = True
        # Reusing deltas not supported for RNNs yet
        self.reuse_deltas = False
        super(RNN, self).__init__(**kwargs)
        req_param(self, ['unrolls'])
        self.rec_layer = self.layers[1]
        opt_param(self, ['num_grad_params'], None)

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
        if (self.num_grad_params is not None) \
                and (str(ensure_dtype(self.backend_type)) ==
                     "<type 'numpy.float64'>"):
            self.grad_checker(numgrad=self.num_grad_params)
        logger.info('commencing model fitting')
        errorlist = []
        suberrorlist = []
        suberror = self.backend.zeros((1, 1))
        while self.epochs_complete < self.num_epochs:
            self.backend.begin(Block.epoch, self.epochs_complete)
            error.fill(0.0)
            mb_id = 1
            self.data_layer.reset_counter()
            dlnb = self.data_layer.num_batches
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
                    logger.info('%d:%d logloss=%0.5f', self.epochs_complete,
                                mb_id / self.step_print,
                                float(error.asnumpyarray()) / dlnb)
                self.backend.end(Block.minibatch, mb_id)
                mb_id += 1
            self.epochs_complete += 1
            errorlist.append(float(error.asnumpyarray()) / dlnb)
            logger.info('epoch: %d, total training error: %0.5f',
                        self.epochs_complete,
                        float(error.asnumpyarray()) / dlnb)
            self.backend.end(Block.epoch, self.epochs_complete - 1)
            self.save_snapshot()
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
              num=None):
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
            if num and num['target'] and (tau == eps_tau):
                # inject epsilon for numerical gradient
                numpy_target = num['target'][num['i'], num['j']].asnumpyarray()
                num['target'][num['i'], num['j']] = (numpy_target + eps)
            if debug:
                logger.debug("in RNNB.fprop, tau %d, input %s" % (tau,
                             inputs[tau].asnumpyarray().argmax(0)[0:5]))
            self.rec_layer.fprop(y[tau-1], c[tau-1], inputs[tau], tau)
            if num and num['target'] and (tau == eps_tau):
                # remove epsilon
                num['target'][num['i'], num['j']] = numpy_target

        # loop for class_layer
        for tau in range(0, self.unrolls):
            if num and num['target'] and (tau == eps_tau):
                # inject epsilon for numerical gradient
                numpy_target = num['target'][num['i'], num['j']].asnumpyarray()
                num['target'][num['i'], num['j']] = (numpy_target + eps)
            if debug:
                logger.debug("in RNNB.fprop, tau %d, input %s" % (tau,
                             inputs[tau].asnumpyarray().argmax(0)[0:5]))
            self.class_layer.fprop(y[tau], tau)
            if num and num['target'] and (tau == eps_tau):
                # remove epsilon
                num['target'][num['i'], num['j']] = numpy_target
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

    def grad_checker(self, numgrad=None):
        """
        Check gradients for LSTM layer:
          - W is replicated, only inject the eps once, repeat, average.
            bProp is only through the full stack, but wrt. the W in each
            level. bProp does this through a for t in tau.

            Need a special fprop that injects into one unrolling only.
        """
        for layer in self.layers:
            logger.info("%s", str(layer))

        num = {'target': None, 'i': 0, 'j': 0}

        if numgrad['name'] == "output":
            num['target'] = self.class_layer.weights
            anl_target = self.class_layer.weight_updates
            num['i'], num['j'] = numgrad['u'], numgrad['v']
        elif numgrad['name'] == "input":
            num['target'] = self.rec_layer.weights
            anl_target = self.rec_layer.weight_updates
            num['i'], num['j'] = numgrad['x'], numgrad['y']
        elif numgrad['name'] == "rec":
            num['target'] = self.rec_layer.weights_rec
            anl_target = self.rec_layer.Wh_updates
            num['i'], num['j'] = numgrad['x'], numgrad['z']

        elif numgrad['name'] == "lstm_fx":
            num['target'] = self.rec_layer.Wfx
            anl_target = self.rec_layer.Wfx_updates
            num['i'], num['j'] = numgrad['x'], numgrad['y']
        elif numgrad['name'] == "lstm_ih":
            num['target'] = self.rec_layer.Wih
            anl_target = self.rec_layer.Wih_updates
            num['i'], num['j'] = numgrad['x'], numgrad['w']
        elif numgrad['name'] == "lstm_fh":
            num['target'] = self.rec_layer.Wfh
            anl_target = self.rec_layer.Wfh_updates
            num['i'], num['j'] = numgrad['x'], numgrad['w']
        elif numgrad['name'] == "lstm_oh":
            num['target'] = self.rec_layer.Woh
            anl_target = self.rec_layer.Woh_updates
            num['i'], num['j'] = numgrad['x'], numgrad['w']
        elif numgrad['name'] == "lstm_ch":
            num['target'] = self.rec_layer.Wch
            anl_target = self.rec_layer.Wch_updates
            num['i'], num['j'] = numgrad['x'], numgrad['w']
        else:
            logger.error("No such numgrad target: '%s'", numgrad['name'])
            raise AttributeError

        eps = numgrad['eps']
        numerical = 0
        #  loop to inject epsilon in different unrolling stages
        for eps_tau in range(0, self.unrolls):
            self.reset(1)  # clear hidden input
            self.fprop(debug=False, eps_tau=eps_tau, eps=0, num=num)
            self.cost_layer.set_targets()
            self.data_layer.reset_counter()
            self.cost_layer.cost.set_outputbuf(
                self.class_layer.output_list[-1])
            suberror_eps = self.cost_layer.get_cost().asnumpyarray()
            self.reset(1)
            self.fprop(debug=False, eps_tau=eps_tau, eps=eps, num=num)
            self.data_layer.reset_counter()
            self.cost_layer.cost.set_outputbuf(
                self.class_layer.output_list[-1])
            suberror_ref = self.cost_layer.get_cost().asnumpyarray()
            num_part = (suberror_ref - suberror_eps) / eps
            logger.debug("numpart for  eps_tau=%d of %d is %e",
                         eps_tau, self.unrolls, num_part)
            numerical += num_part

        # bprop for analytical gradient
        self.bprop(debug=False, numgrad=numgrad)

        analytical = anl_target[num['i'], num['j']].asnumpyarray()
        logger.debug("--------------------------------------------------")
        logger.debug("Numerical gradient checks: Only fp64 CPU supported")
        logger.debug("RNN grad_checker: suberror_eps %f", suberror_eps)
        logger.debug("RNN grad_checker: suberror_ref %f", suberror_ref)
        logger.debug("RNN grad_checker: numerical %e", numerical)
        logger.debug("RNN grad_checker: analytical %e", analytical)
        logger.debug("RNN grad_checker: ratio %e", 1./(numerical/analytical))
        logger.debug("--------------------------------------------------")

    def predict_generator(self, dataset, setname):
        """
        Generate predicitons and true labels for the given dataset.
        """
        self.data_layer.init_dataset(dataset)
        assert self.data_layer.has_set(setname)
        self.data_layer.use_set(setname, predict=True)
        self.data_layer.reset_counter()

        predlabels = self.backend.empty((1, self.batch_size))
        labels = self.backend.empty((1, self.batch_size))

        outputs_pred = self.backend.zeros((self.unrolls, self.batch_size))
        outputs_targ = self.backend.zeros((self.unrolls, self.batch_size))

        mb_id = 0
        self.data_layer.reset_counter()
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
                outputs_pred[tau, :] = predlabels
                outputs_targ[tau, :] = labels

            yield (outputs_pred.reshape((1, self.unrolls * self.batch_size)),
                   outputs_targ.reshape((1, self.unrolls * self.batch_size)))

        self.data_layer.cleanup()

    def predict_fullset(self, dataset, setname):
        """
        Generate predicitons and true labels for the given dataset.
        Note that this requires enough memory to house the predictions and
        labels for the entire dataset at one time (not recommended for large
        datasets, see predict_generator instead).

        Agruments:
            dataset: A neon dataset instance
            setname: Which set to compute predictions for (test, train, val)

        Returns:
            tuple: on each call will yield a 2-tuple of outputs and references.
                   The first item is the model probabilities for each class,
                   and the second item is either the one-hot or raw labels with
                   ground truth.

        See Also:
            predict_generator
        """
        self.data_layer.init_dataset(dataset)
        assert self.data_layer.has_set(setname)
        self.data_layer.use_set(setname, predict=True)
        self.data_layer.reset_counter()

        predlabels = self.backend.empty((1, self.batch_size))
        labels = self.backend.empty((1, self.batch_size))

        outputs_pred = self.backend.zeros((self.data_layer.num_batches *
                                           self.unrolls, self.batch_size))
        outputs_targ = self.backend.zeros((self.data_layer.num_batches *
                                           self.unrolls, self.batch_size))

        mb_id = 0
        self.data_layer.reset_counter()
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

        self.write_string(pred_flat, targ_flat, setname)

        return (pred_flat, targ_flat)

    def write_string(self, pred, targ, setname):
            """
            For text prediction, reassemble the batches and print out a short
            contigous segment of target text and predicted text - useful to
            check for off-by-one errors and the like.

            Note: This is a debug function, it's not called anywhere by default
            """
            import numpy as np

            pred_int = pred[0, 100:140].asnumpyarray().ravel().astype(np.int8) \
                + 31
            targ_int = targ[0, 100:140].asnumpyarray().ravel().astype(np.int8) \
                + 31
            # remove special characters, replace them with '#'
            pred_int[pred_int < 32] = 35
            targ_int[targ_int < 32] = 35

            # create output strings
            logging.info("the target for '%s' is: '%s'", setname,
                         ''.join(targ_int.view('c')))
            logging.info("prediction for '%s' is: '%s'", setname,
                         ''.join(pred_int.view('c')))
            logging.info("successes for '%s' are: '%s'", setname,
                         ''.join((88 * (targ_int == pred_int) + 32).view('c')))
