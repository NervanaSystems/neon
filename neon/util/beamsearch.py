import numpy as np
from neon import NervanaObject
import logging


logger = logging.getLogger(__name__)


class BeamSearch(NervanaObject):
    """
    Beam search for Encoder-Decoder models.

    Arguments:
        seq2seq (Object): Seq2Seq container object with a trained model to use for inference
    """
    def __init__(self, seq2seq):
        super(BeamSearch, self).__init__(name=None)

        self.layers = seq2seq

        self.hasLUT = seq2seq.hasLUT

        # resize buffers
        new_steps = 1
        seq2seq.decoder.switch_mode(inference=True)
        self.z_shape = new_steps if self.hasLUT else (seq2seq.out_shape[0], new_steps)

    def beamsearch(self, inputs, num_beams=5, steps=None):
        """
        Perform an fprop path and beam search on a given set of network inputs.

        Arguments:
            inputs (Tensor): Minibatch of network inputs
            num_beams (Int): Number of beams (hypothesis) to search over
            steps (Int): Length of desired output in number of time steps
        """
        self.num_beams = num_beams

        self.num_live = 1
        self.num_dead = 0

        # allocate space for outputs, hypotheses and associated scores
        # default to the number of steps used by this decoder most recently.
        if steps is None:
            steps = self.layers.in_shape[1]

        bsz = self.be.bsz

        self.z_list = [self.be.iobuf(self.z_shape) for _ in range(num_beams)]
        for i in range(num_beams):
            if getattr(self.layers.decoder, 'start_index', None) is not None:
                self.z_list[i][self.layers.decoder.start_index] = 1

        self.candidates = [np.zeros((steps, bsz)) for _ in range(num_beams)]
        self.scores = [np.zeros(bsz) for _ in range(num_beams)]

        z = self.be.iobuf(self.z_shape)

        # encoder
        self.layers.encoder.fprop(inputs, inference=True, beta=0.0)
        final_states = self.layers.encoder.get_final_states(
            self.layers.decoder_connections
        )

        if len(final_states) != len(self.layers.decoder._recurrent):
            raise ValueError((
                'number of decoder layers ({num_layers}) does not match '
                'the number of decoder connections ({num_decoder_connections}).'
            ).format(
                num_layers=len(self.layers.decoder._recurrent),
                num_decoder_connections=len(final_states),
            ))
        else:
            init_state_list = final_states

        # num_beams x num_layers x layer_shape
        self.init_state_lists = [[self.be.zeros_like(rec.h[-1])
                                 for rec in self.layers.decoder._recurrent]
                                 for _ in range(num_beams)]  # TODO: rename
        self.init_state_lists[0] = init_state_list

        # decoder
        if self.hasLUT:
            z_beams = [self.be.iobuf(self.layers.decoder.out_shape) for _ in range(self.num_beams)]
        else:
            z_beams = [self.be.iobuf(self.z_shape) for _ in range(self.num_beams)]

        # num_beams x num_layers x layer_shape
        hidden_state_beams = [[self.be.zeros_like(l.final_state())
                               for l in self.layers.decoder._recurrent]
                              for _ in range(self.num_beams)]  # will be list of lists

        for t in range(steps):
            # fprop all beams
            for bb in range(self.num_live):
                z = self.layers.decoder.fprop(self.z_list[bb], inference=True,
                                              init_state_list=self.init_state_lists[bb])
                z_beams[bb][:] = z
                for i, l in enumerate(self.layers.decoder._recurrent):
                    hidden_state_beams[bb][i][:] = l.final_state()
            # search over all beams
            self._beamsearch_step(t, z_beams, hidden_state_beams)

        return self.be.array(self.candidates[-1])

    def _beamsearch_step(self, t, z_list_t, init_state_lists):
        """
        Arguments:
            t (int): time step
            z_list (list of tensors) : fprop outputs for all beams
        """
        num_out = self.layers.out_shape[0]

        # bring init_state_lists list of lists over to CPU
        # init_state_lists_in, init_state_lists_out
        init_state_lists_in = []
        for isl in init_state_lists:
            init_state_lists_in.append([state.get() for state in isl])

        # num_beams x num_layers x layer_hidden_shape
        init_state_lists_out = [[np.zeros(tnsr.shape) for tnsr in init_state_lists[0]]
                                for _ in range(len(init_state_lists_in))]

        # pick highest scoring across all beams
        z_list = [z_list_t[bb].get() for bb in range(self.num_live)]
        scores_list = [np.log(z) + score for z, score in zip(z_list, self.scores)]
        scores = np.vstack(scores_list)
        self.num_live = num_live = self.num_beams - self.num_dead
        ind = scores.argsort(axis=0)[-num_live:]  # top scores across both old beams and new tokens
        self.scores_list = scores_list  # for testing

        # store candidates and scores for all beams and samples in batch
        hyp_num = (ind // num_out).astype(np.int32)  # hypothesis number
        word_num = ind % num_out  # onehot word index

        # create a copy of candidates buffer to read from so it does not get overwritten
        old_candidates = [can.copy() for can in self.candidates]

        for bb in range(num_live):
            for hyp in range(len(scores_list)):
                ex_with_hyp = np.where(hyp_num[bb, :] == hyp)[0]
                self.candidates[bb][:, ex_with_hyp] = old_candidates[hyp][:, ex_with_hyp]
                wnum = word_num[bb, ex_with_hyp]
                self.scores[bb][ex_with_hyp] = scores_list[hyp][wnum, ex_with_hyp]
            self.candidates[bb][t, :] = word_num[bb, :]

        # get decoder hidden states for each beam
        for bb in range(num_live):
            for hyp in range(len(scores_list)):
                ex_with_hyp = np.where(hyp_num[bb, :] == hyp)[0]

                # transfer hidden state from DECODER to next step
                for ii in range(len(self.layers.decoder._recurrent)):
                    init_state_lists_out[bb][ii][:, ex_with_hyp] = \
                        init_state_lists_in[hyp][ii][:, ex_with_hyp]

        # get inputs for next step for each beam
        for bb in range(num_live):
            # create input for next time step
            if self.hasLUT:
                # use index. TODO: Backend needs to catch these striding mismatches
                inputs = self.be.array(self.candidates[bb][t, :].reshape(1, -1))
                self.z_list[bb][:] = inputs
            else:
                # convoluted softmax -> index -> onehot scheme
                inputs = self.be.array(self.candidates[bb][t, :], dtype=np.int32)
                self.z_list[bb][:] = self.be.onehot(inputs, axis=0)

        # copy list of lists of numpy matrices into list of lists of GPU tensors
        for ii, isl in enumerate(init_state_lists_out):
            for jj, state in enumerate(isl):
                self.init_state_lists[ii][jj][:] = state
