# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
# ----------------------------------------------------------------------------
import pickle
import numpy as np

from neon.layers import LookupTable, GRU, Affine
from neon.layers.container import Sequential
from neon.initializers import Uniform, Orthonormal, Constant
from neon.transforms import Softmax, Logistic, Tanh


class SkipThought(Sequential):

    """
    A skip-thought container that encapsulates the network architectue

    Arguments:
        vocab_size: vocabulary size
        embed_dim: word embedding dimension
        init_embed: word embedding initialization
        nhidden: number of hidden units
    """

    def __init__(self, vocab_size, embed_dim, init_embed, nhidden, name=None):

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.init_embed = init_embed
        self.nhidden = nhidden
        self.owns_output = True
        self.owns_delta = True

        init_rec = Orthonormal()
        init_ff = Uniform(low=-0.1, high=0.1)

        self.embed_s = LookupTable(vocab_size=vocab_size, embedding_dim=embed_dim,
                                   init=init_embed, pad_idx=0)
        self.embed_p = LookupTable(vocab_size=vocab_size, embedding_dim=embed_dim,
                                   init=init_embed, pad_idx=0)
        self.embed_n = LookupTable(vocab_size=vocab_size, embedding_dim=embed_dim,
                                   init=init_embed, pad_idx=0)

        self.encoder = GRU(nhidden, init=init_ff, init_inner=init_rec, activation=Tanh(),
                           gate_activation=Logistic(), reset_cells=True)
        self.decoder_p = GRU(nhidden, init=init_ff, init_inner=init_rec, activation=Tanh(),
                                     gate_activation=Logistic(), reset_cells=True)
        self.decoder_n = GRU(nhidden, init=init_ff, init_inner=init_rec, activation=Tanh(),
                                     gate_activation=Logistic(), reset_cells=True)

        self.affine_p = Affine(
            vocab_size, init=init_ff, bias=Constant(0.), activation=Softmax())
        self.affine_n = Affine(
            vocab_size, init=init_ff, bias=Constant(0.), activation=Softmax())

        self.layers = [self.embed_s, self.embed_p, self.embed_n, self.encoder,
                       self.decoder_p, self.decoder_n] + self.affine_p + self.affine_n

        self.layer_dict = dict()
        self.layer_dict['lookupTable'] = self.embed_s
        self.layer_dict['encoder'] = self.encoder
        self.layer_dict['decoder_previous'] = self.decoder_p
        self.layer_dict['decoder_next'] = self.decoder_n
        self.layer_dict['affine'] = self.affine_p

        super(SkipThought, self).__init__(layers=self.layers, name=name)

    def allocate(self, shared_outputs=None):
        super(SkipThought, self).allocate(shared_outputs)
        self.error_ctx = self.be.iobuf((self.nhidden, self.encoder.nsteps))

        self.dW_embed = self.be.empty_like(self.embed_s.dW)
        self.dW_linear = self.be.empty_like(self.affine_p[0].dW)
        self.dW_bias = self.be.empty_like(self.affine_p[1].dW)

        # two affine layers are init differently, need to syn the weights
        self.affine_p[0].W[:] = self.affine_n[0].W

    def configure(self, in_obj):
        """
        in_obj should be one single input shape as all three sentences will go through the
        word embedding layer first
        """
        if not isinstance(in_obj, list):
            in_obj = in_obj.shape
        self.in_shape = in_obj
        self.embed_s.configure(in_obj[0])
        self.embed_s.set_next(self.encoder)
        self.encoder.configure(self.embed_s)

        self.embed_p.configure(in_obj[1])
        self.embed_p.set_next(self.decoder_p)
        self.decoder_p.configure(self.embed_p)
        self.decoder_p.set_next(self.affine_p)
        prev_in = self.decoder_p
        for l in self.affine_p:
            l.configure(prev_in)
            prev_in.set_next(l)
            prev_in = l

        self.embed_n.configure(in_obj[2])
        self.embed_n.set_next(self.decoder_n)
        self.decoder_n.configure(self.embed_n)
        self.decoder_n.set_next(self.affine_n)
        prev_in = self.decoder_n
        for l in self.affine_n:
            l.configure(prev_in)
            prev_in.set_next(l)
            prev_in = l

        self.out_shape = [
            self.affine_p[-1].out_shape, self.affine_n[-1].out_shape]

        return self

    def fprop(self, inputs, inference=False, beta=0.0):
        """
        TODO: implement the inference part
        """
        assert len(inputs) == 3

        s_sent = inputs[0]
        p_sent = inputs[1]
        n_sent = inputs[2]

        # process the source sentence
        emb_s = self.embed_s.fprop(s_sent, inference)
        enc_s = self.encoder.fprop(emb_s, inference)
        context_state = enc_s[:, -self.be.bsz:]

        # process the previous sentence
        emb_p = self.embed_p.fprop(p_sent, inference)
        dec_p = self.decoder_p.fprop(emb_p, inference=inference, init_state=context_state)
        x = dec_p
        for l in self.affine_p:
            x = l.fprop(x, inference)
        aff_p = x

        # process the next sentence
        emb_n = self.embed_n.fprop(n_sent, inference)
        dec_n = self.decoder_n.fprop(emb_n, inference=inference, init_state=context_state)
        x = dec_n
        for l in self.affine_n:
            x = l.fprop(x, inference)
        aff_n = x

        return [aff_p, aff_n]

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        error should have 2 elements

        TODO:
              2. make sure the error bproped to encoder is correct
        """
        assert len(error) == 2

        error_p = error[0]
        error_n = error[1]

        for l in reversed(self.affine_p):
            error_p = l.bprop(error_p)
        error_p = self.decoder_p.bprop(error_p)
        error_ctx_p = self.decoder_p.h_delta[0]
        error_p = self.embed_p.bprop(error_p)

        for l in reversed(self.affine_n):
            error_n = l.bprop(error_n)
        error_n = self.decoder_n.bprop(error_n)
        error_ctx_n = self.decoder_n.h_delta[0]
        error_n = self.embed_n.bprop(error_n)

        self.error_ctx.fill(0)
        self.error_ctx[:, -self.be.bsz:] = error_ctx_p + error_ctx_n

        error_s = self.encoder.bprop(self.error_ctx)
        error_s = self.embed_s.bprop(error_s)

        # sync the three embedding layers' dW
        self.dW_embed[:] = (
            self.embed_s.dW + self.embed_p.dW + self.embed_n.dW)/3
        self.embed_s.dW[:] = self.dW_embed
        self.embed_p.dW[:] = self.dW_embed
        self.embed_n.dW[:] = self.dW_embed

        # sync the two affine layers' dW
        self.dW_linear[:] = (self.affine_p[0].dW + self.affine_n[0].dW)/2
        self.affine_p[0].dW[:] = self.dW_linear
        self.affine_n[0].dW[:] = self.dW_linear
        self.dW_bias[:] = (self.affine_p[1].dW + self.affine_n[1].dW)/2
        self.affine_p[1].dW[:] = self.dW_bias
        self.affine_n[1].dW[:] = self.dW_bias

        return error_s

    def get_terminal(self):
        terminal = [
            self.affine_p[-1].get_terminal(), self.affine_n[-1].get_terminal()]
        return terminal



