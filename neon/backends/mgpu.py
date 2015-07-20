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
Neon multigpu backend for NervanaGPU.
Decorates most functions from the single GPU backend.
The basic tensor type used here is the Multi GPU Tensor, which tracks shards
and replicas of data, weights, etc on different devices via maintenance of
contexts and an internal tensor list.
"""
import logging

from neon.backends.backend import Block
from neon.backends.gpu import GPU
from nervanagpu import NervanaGPU, GPUTensor
import pycuda.driver as drv
import numpy as np
from functools import wraps
import atexit

logger = logging.getLogger(__name__)


def replicate(method):
    def decorate(cls):
        @wraps(cls)
        def func(self, *args, **kwargs):
            if self.ng.block is not None:
                self.call_stack.append((method, args, kwargs))
                return
            else:
                tsrlist = []
                for idx, ctx in enumerate(getattr(self, 'ctxs')):
                    ctx.push()
                    self.ng.stream = self.strms[idx]
                    myargs = [a.tlist[idx] if isinstance(
                        a, MGPUTensor) else a for a in args]
                    mykwargs = {k: v.tlist[idx] if isinstance(
                        v, MGPUTensor) else v for k, v in kwargs.iteritems()}
                    tsrlist.append(
                        getattr(super(cls, self), method)(*myargs, **mykwargs))
                    self.ng.stream = None
                    ctx.pop()
                return MGPUTensor(tsrlist) if tsrlist[0] is not None else None
        setattr(cls, method, func)
        return cls
    return decorate


def passthru(method):
    def decorate(cls):
        @wraps(cls)
        def func(self, *args, **kwargs):
            tsrlist = []
            for idx, (tsr, ctx) in enumerate(zip(getattr(self, '_tensorlist'),
                                                 getattr(self, 'ctxs'))):
                ctx.push()
                myargs = [a.tlist[idx] if isinstance(
                    a, MGPUTensor) else a for a in args]
                mykwargs = {k: v.tlist[idx] if isinstance(
                    v, MGPUTensor) else v for k, v in kwargs.iteritems()}
                tsrlist.append(getattr(tsr, method)(*myargs, **mykwargs))
                ctx.pop()
            if tsrlist[0] is not None:
                return MGPUTensor(tsrlist, ptype=self.ptype)

        setattr(cls, method, func)
        return cls
    return decorate


@passthru('_assign')
@passthru('fill')
@passthru('reshape')
@passthru('copy_from')
@passthru('__getitem__')
@passthru('__add__')
@passthru('__sub__')
@passthru('__mul__')
@passthru('__div__')
@passthru('__truediv__')
@passthru('__pow__')
@passthru('__radd__')
@passthru('__rsub__')
@passthru('__rmul__')
@passthru('__rdiv__')
@passthru('__ne__')
@passthru('__eq__')
class MGPUTensor(object):
    ctxs = None
    num_dev = 0

    def __init__(self, tensorlist, ptype='fragment'):
        self._tensorlist = tensorlist
        self.ptype = ptype

    @property
    def shape(self):
        return self._tensorlist[0].shape

    @property
    def dtype(self):
        return self._tensorlist[0].dtype

    @property
    def size(self):
        return self._tensorlist[0].size

    @property
    def is_contiguous(self):
        return self._tensorlist[0].is_contiguous

    @property
    def tlist(self):
        return self._tensorlist

    @property
    def ptr(self):
        return self._tensorlist[0].gpudata.__int__()

    def __setitem__(self, index, value):
        if self.ctxs is None:
            raise ValueError("Contexts not defined")
        for idx, (tsr, ctx) in enumerate(zip(getattr(self, '_tensorlist'),
                                             getattr(self, 'ctxs'))):
            ctx.push()
            if isinstance(value, MGPUTensor):
                tsr.__setitem__(index, value._tensorlist[idx])
            else:
                tsr.__setitem__(index, value)
            ctx.pop()

    def asnumpyarray(self):
        if self.ptype == 'replica':
            self.ctxs[0].push()
            rval = self._tensorlist[0].get()
            self.ctxs[0].pop()
            return rval
        else:
            rval = []
            for subtensor, ctx in zip(self.tlist, self.ctxs):
                ctx.push()
                npv = subtensor.get()
                rval.append(npv)
                ctx.pop()
            if self.ptype == 'vfragment':
                return np.vstack(rval)
            else:
                return np.hstack(rval)

    @property
    def T(self):  # noqa
        """
        return a transposed view
        """
        tsrlist = []
        for tsr in self._tensorlist:
            tsrlist.append(GPUTensor(backend=tsr.backend,
                                     shape=tsr.shape[::-1], dtype=tsr.dtype,
                                     allocator=tsr.allocator, base=tsr,
                                     gpudata=tsr.gpudata,
                                     strides=tsr.strides[::-1],
                                     is_trans=(not tsr.is_trans),
                                     name=tsr.name, rounding=tsr.rounding))
        return self.__class__(tsrlist)


@replicate('fprop_conv')
@replicate('convolution')
@replicate('bprop_conv')
@replicate('update_conv')
@replicate('fprop_pool')
@replicate('bprop_pool')
@replicate('logistic')
@replicate('rectlin')
@replicate('rectlin_derivative')
@replicate('rectleaky')
@replicate('rectleaky_derivative')
@replicate('sum')
@replicate('mean')
@replicate('min')
@replicate('max')
@replicate('variance')
@replicate('fabs')
@replicate('sqrt')
@replicate('zeros')
@replicate('ones')
@replicate('empty')
@replicate('array')
@replicate('add')
@replicate('subtract')
@replicate('multiply')
@replicate('divide')
@replicate('greater')
@replicate('equal')
@replicate('not_equal')
@replicate('clip')
@replicate('log')
@replicate('tanh')
@replicate('argmax')
@replicate('softmax')
@replicate('softmax_gradient')
@replicate('make_binary_mask')
@replicate('gdm_compound')
@replicate('gdmwd_compound')
@replicate('ada_update')
@replicate('crossent')
@replicate('transpose')
@replicate('logistic_compound')
@replicate('fprop_bn_compound')
@replicate('bprop_bn_compound')
class MGPU(GPU):
    default_dtype = np.float32
    num_dev = 1
    is_dist = True

    def __init__(self, rng_seed, stochastic_round=False, device_id=0,
                 num_dev=2):
        drv.init()
        self.num_dev = num_dev

        if device_id == 0:
            self.dev_list = range(num_dev)
        else:
            self.dev_list = device_id

        assert len(self.dev_list) == self.num_dev
        assert self.num_dev <= drv.Device.count()

        self.ctxs = []
        self.devs = []
        self._strms = []
        self._redstrms = []

        self._events = []
        self._redevents = []

        self.async = True
        self._nostrms = [None for i in self.dev_list]

        for i in self.dev_list:
            self.devs.append(drv.Device(i))

        for dev in self.devs:
            self.ctxs.append(
                dev.make_context(drv.ctx_flags.SCHED_BLOCKING_SYNC))
            self._strms.append(drv.Stream())
            self._redstrms.append(drv.Stream())
            self._events.append(drv.Event())
            self._redevents.append(drv.Event())
            drv.Context.pop()

        self.ctxs[0].push()
        atexit.register(drv.Context.pop)
        MGPUTensor.ctxs = self.ctxs
        MGPUTensor.num_dev = num_dev

        self.ng = NervanaGPU(stochastic_round=stochastic_round)
        logger.info("Initialized %d device NervanaGPU, stochastic_round=%s",
                    num_dev, stochastic_round)
        self.ng.block = None
        self.rng_seed = rng_seed
        self.rng_init()

        # Setup the pairwise contexts
        # TODO clean up this code to avoid indexing
        for dev1, ctx1 in zip(self.devs, self.ctxs):
            ctx1.push()
            for dev2, ctx2 in zip(self.devs, self.ctxs):
                if dev1 == dev2:
                    continue
                if dev1.can_access_peer(dev2):
                    ctx1.enable_peer_access(ctx2)
                else:
                    print('Cannot enable peer access between '
                          '{:d} and {:d}'.format(dev1, dev2))
            ctx1.pop()

    def make_events(self):
        evtlist = []
        for ctx in self.ctxs:
            ctx.push()
            evtlist.append(drv.Event())
            ctx.pop()
        return evtlist

    # These definitions are for performing grouped context commands
    # This is experimental and should remove _stack for actual usage
    def begin_stack(self, block, identifier):
        if block == Block.update:
            self.ng.block = Block.update
            self.call_stack = []
        else:
            pass

    def end_stack(self, block, identifier):
        if block == Block.update:
            self.ng.block = None
            for idx, ctx in enumerate(self.ctxs):
                ctx.push()
                self.ng.stream = self.strms[idx]
                for method, args, kwargs in self.call_stack:
                    myargs = [a._tensorlist[idx] if isinstance(
                        a, MGPUTensor) else a for a in args]
                    mykwargs = {k: v._tensorlist[idx] if isinstance(
                        v, MGPUTensor) else v for k, v in kwargs.iteritems()}
                    getattr(super(MGPU, self), method)(*myargs, **mykwargs)
                self.ng.stream = None
                ctx.pop()
            self.call_stack = None
        else:
            pass

    @property
    def strms(self):
        return self._strms if self.async else self._nostrms

    @property
    def redstrms(self):
        return self._redstrms if self.async else self._nostrms

    def uniform(self, low=0.0, high=1.0, size=1, dtype=default_dtype,
                name=None, persist_values=True, ptype='replica'):
        """
        generate numpy random number and convert to a GPUTensor.
        If called with dtype=None it will probably explode
        """
        assert len(size) == 2
        result = self.empty(size, dtype=dtype, persist_values=persist_values)
        result.ptype = ptype
        beshape = size if ptype == 'replica' else (self.num_dev * size[0],
                                                   size[1])
        ary = np.random.uniform(low, high, beshape).astype(dtype)
        self.set(result, ary)
        return result

    def normal(self, loc=0.0, scale=1.0, size=1, dtype=default_dtype,
               name=None, persist_values=True, ptype='replica'):
        """
        Gaussian/Normal random number sample generation
        """
        assert len(size) == 2
        result = self.empty(size, dtype=dtype, persist_values=persist_values)
        result.ptype = ptype
        beshape = size if ptype == 'replica' else (self.num_dev * size[0],
                                                   size[1])
        ary = np.random.normal(loc, scale, beshape).astype(dtype)
        self.set(result, ary)
        return result

    def synchronize(self):
        if not self.async:
            return
        for s in self.strms:
            s.synchronize()

    def redsynchronize(self):
        if not self.async:
            return
        for s in self.redstrms:
            s.synchronize()

    def allocate_fragment(self, shape, dtype=default_dtype,
                          persist_values=True):
        # TODO: set ptype to be fragment in this case ??
        return self.empty((shape[0], shape[1] / self.num_dev), dtype,
                          persist_values=persist_values)

    def zeros_like(self, ary, dtype=default_dtype, persist_values=True,
                   name=None):
        result = self.zeros(ary.shape, dtype=dtype,
                            persist_values=persist_values)
        result.ptype = ary.ptype
        return result

    def empty_like(self, ary, dtype=default_dtype, persist_values=True,
                   name=None):
        result = self.empty(ary.shape, dtype=dtype,
                            persist_values=persist_values, name=name)
        result.ptype = ary.ptype
        return result

    def set(self, tensor, data):
        assert isinstance(tensor, MGPUTensor)
        if tensor.ptype == 'replica':
            for dest, strm, ctx in zip(tensor.tlist, self.strms, self.ctxs):
                ctx.push()
                drv.memcpy_htod_async(dest.ptr, data, strm)
                ctx.pop()
            # tensor.copy_from(data)
        else:
            self.scatter(data, tensor)

    def scatter(self, hbuf, dbuf):
        '''
        scatters the array data in hbuf to the mgpu tensor
        assumes that dbuf is a M x N and hbuf is M x (Nxk) where k is the
        number of replicas
        also assumes that dtype of hbuf and dbuf are the same
        '''
        assert hbuf.size == dbuf.size * dbuf.num_dev
        assert isinstance(dbuf, MGPUTensor)
        assert hbuf.dtype == dbuf.dtype
        ndata = dbuf.size
        starts = [i * ndata for i in range(self.num_dev)]

        for dest, strm, ctx, doff in zip(dbuf.tlist, self.strms, self.ctxs,
                                         starts):
            src = hbuf.reshape((hbuf.size))[doff:(doff + ndata)]
            ctx.push()
            drv.memcpy_htod_async(dest.ptr, src, strm)
            ctx.pop()

        self.synchronize()

    def fprop_fc(self, out, inputs, weights, layer=None):
        """
        In this case, the weights are shards, the acts are replicas
        ubuf should be of size nout/num_dev x mbsz
        """
        ubuf = layer.mempool[0]
        assert ubuf.shape == (weights.shape[0], inputs.shape[1])

        if layer.use_biases:
            biases = layer.biases.tlist
        else:
            biases = [None for i in range(self.num_dev)]

        for dbuf, ibuf, wt, bs, strm, ctx in zip(ubuf.tlist, inputs.tlist,
                                                 weights.tlist, biases,
                                                 self.strms, self.ctxs):
            ctx.push()
            self.ng.stream = strm
            self.ng.dot(wt, ibuf, dbuf)
            if layer.use_biases:
                self.ng.add(dbuf, bs, out=dbuf)
            ctx.pop()

        # Note, should be safe not to sync because each fragment is computed
        # on the same stream that originates the copy
        # self.synchronize()
        self.fragment_to_replica(ubuf, out)

    def bprop_fc(self, out, weights, deltas, layer=None):
        """
        Backward propagate the error through a fully connected network layer.

        Arguments:
            out (GPUTensor): Where to store the backward propagated errors.
            weights (GPUTensor): The weight coefficient values for this layer.
            deltas (GPUTensor): The error values for this layer
            layer (Layer): The layer object.
        """
        ubuf = layer.mempool[1]
        wtsz = weights.shape[0]
        starts = [i * wtsz for i in range(self.num_dev)]
        assert out.shape == (weights.shape[1], deltas.shape[1])
        assert ubuf.shape == out.shape

        for dbuf, ibuf, wt, strm, ctx, off in zip(out.tlist, deltas.tlist,
                                                  weights.tlist, self.strms,
                                                  self.ctxs, starts):
            ctx.push()
            self.ng.stream = strm
            self.ng.dot(wt.T, ibuf[off:(off + wtsz)], dbuf)
            ctx.pop()

        # Note, should be safe not to sync because each fragment is computed
        # on the same stream that originates the copy
        self.synchronize()
        self.reduce(out, ubuf)

    def update_fc(self, out, inputs, deltas, layer=None):
        wtsz = out.shape[0]
        starts = [i * wtsz for i in range(self.num_dev)]

        for obuf, dbuf, ibuf, strm, ctx, off in zip(out.tlist, deltas.tlist,
                                                    inputs.tlist, self.strms,
                                                    self.ctxs, starts):
            ctx.push()
            self.ng.stream = strm
            self.ng.dot(dbuf[off:(off + wtsz)], ibuf.T, obuf)
            ctx.pop()

        # self.synchronize()

    def update_fc_bias(self, err, out):
        """
        Compute the updated bias gradient for a fully connected network layer.

        Arguments:
            out (GPUTensor): Where to store the updated gradient value.
            err (GPUTensor): backpropagated error
        """
        wtsz = out.shape[0]
        starts = [i * wtsz for i in range(self.num_dev)]

        for ebuf, obuf, strm, ctx, off in zip(err.tlist, out.tlist, self.strms,
                                              self.ctxs, starts):
            ctx.push()
            self.ng.stream = strm
            self.ng.sum(ebuf[off:(off + wtsz)], axis=1, out=obuf)
            ctx.pop()

    def add_fc_bias(self, inputs, bias):
        """
        This is a no-op since we absorb the bias add into the fprop_fc call
        """
        pass

    def reduce_tensor(self, ary, async=True):
        '''
        This is the case for the scalar tensor
        '''
        assert ary.size == 1
        if ary.ptype == 'replica':
            self.ctxs[0].push()
            result = ary.tlist[0].get()
            self.ctxs[0].pop()
            return result

        result = np.zeros((self.num_dev, 1), ary.dtype)
        for i, (ctx, src_buf, strm) in enumerate(zip(
                self.ctxs, ary.tlist, self.strms)):
            ctx.push()
            drv.memcpy_dtoh_async(result[i], src_buf.ptr, strm)
            ctx.pop()
        self.synchronize()
        return result.sum()

    def replica_to_fragment(self, reptsr, fragtsr):
        '''
        Scatters the replica into the fragments (this just discards, so no p2p
        communication necessary
        '''
        numrep = self.num_dev
        fragsz = fragtsr.size
        dsz = fragtsr.dtype.itemsize
        assert reptsr.size == fragsz * numrep
        strms = self.strms
        starts = [i * fragsz for i in range(numrep)]

        for dbuf, sbuf, ctx, offset, strm in zip(fragtsr.tlist, reptsr.tlist,
                                                 self.ctxs, starts, strms):
            ctx.push()
            drv.memcpy_dtod_async(dbuf.ptr, sbuf.ptr + offset * dsz,
                                  fragsz * dsz, strm)
            ctx.pop()

        self.synchronize()

    def fragment_to_replica(self, fragtsr, reptsr):
        '''
        Gathers the fragments from fragtsr into reptsr
        '''
        numrep = self.num_dev
        fragsz = fragtsr.size
        dsz = fragtsr.dtype.itemsize
        assert reptsr.size == fragsz * numrep
        assert fragtsr.is_contiguous
        starts = [i * fragsz for i in range(numrep)]

        for dbuf, dctx in zip(reptsr.tlist, self.ctxs):
            for sbuf, sctx, soff, strm in zip(fragtsr.tlist, self.ctxs,
                                              starts, self.strms):
                myargs = [dbuf.ptr + soff * dsz, sbuf.ptr, fragsz * dsz]
                if sctx == dctx:
                    cpfunc = drv.memcpy_dtod_async
                else:
                    cpfunc = drv.memcpy_peer_async
                    myargs.extend([dctx, sctx])
                myargs.append(strm)
                sctx.push()
                cpfunc(*myargs)
                sctx.pop()

        self.synchronize()

    def share_activations(self, in_acts, out_acts, tmpbufs):
        """
        Not ideal we have to do this, but for now this is enough to get things
        working
        Placeholder function to deal with the facts that we lay our tensors
        out in column-major form.  Fast inplace transpose can be placed in
        here for improved performance
        in_acts:  fragment tensor to be joined
        out_acts: replica tensor to receive
        tmpbufs:  contains temporary storage for the transposes
        """
        assert in_acts.shape[0] == out_acts.shape[0]
        assert in_acts.shape[1] == out_acts.shape[1] / self.num_dev
        (rbufT, fbufT) = tmpbufs
        assert fbufT.shape == in_acts.shape[::-1]
        assert rbufT.shape == out_acts.shape[::-1]

        self.transpose(in_acts, fbufT)
        self.fragment_to_replica(fbufT, rbufT)
        self.transpose(rbufT, out_acts)

    def split_activations(self, in_acts, out_acts, tmpbufs):
        """
        Not ideal we have to do this, but for now this is enough to get things
        working
        Placeholder function to deal with the facts that we lay our tensors
        out in column-major form.  Fast inplace transpose can be placed in
        here for improved performance
        in_acts:  replica tensor to be split into fragments
        out_acts: fragment tensor to receive
        tmpbufs:  contains temporary storage for the transposes
        """
        assert in_acts.shape[0] == out_acts.shape[0]
        assert in_acts.shape[1] == out_acts.shape[1] * self.num_dev
        (rbufT, fbufT) = tmpbufs
        assert fbufT.shape == out_acts.shape[::-1]
        assert rbufT.shape == in_acts.shape[::-1]

        self.transpose(in_acts, rbufT)
        self.replica_to_fragment(rbufT, fbufT)
        self.transpose(fbufT, out_acts)

    def reduce(self, ary, ubuf):
        """
        Does a summation reduction of the fragments in ary and rebroadcasts
        them into ary using butterfly reduction.  Uses ubuf as a temporary
        storage buffer.

        Requires that ubuf has same dtype as ary, and the size of ubuf is large
        enough to store the broadcasted sub-fragments of ary.
        """
        numrep = self.num_dev
        totsz = ary.size
        subsz = (totsz + numrep - 1) / numrep
        dsz = ary.dtype.itemsize
        assert ubuf.size == subsz * numrep
        starts = [i * subsz for i in range(numrep)]
        strmlist = self.redstrms
        evtlist = self._redevents

        # GATHER SUB-FRAGMENTS
        for dbuf, dctx, doff, dstrm, evt in zip(ubuf.tlist, self.ctxs, starts,
                                                strmlist, evtlist):
            for sbuf, sctx, soff, strm in zip(ary.tlist, self.ctxs,
                                              starts, strmlist):
                myargs = [dbuf.ptr + soff * dsz,
                          sbuf.ptr + doff * dsz,
                          min(subsz, totsz - doff) * dsz]
                if sctx == dctx:
                    cpfunc = drv.memcpy_dtod_async
                else:
                    cpfunc = drv.memcpy_peer_async
                    myargs.extend([dctx, sctx])
                myargs.append(strm)
                cpfunc(*myargs)
            evt.record(dstrm)

        # REDUCTION of SUB-FRAGMENTS
        for sbuf, dbuf, sctx, soff, strm, evt in zip(ary.tlist, ubuf.tlist,
                                                     self.ctxs, starts,
                                                     strmlist, evtlist):
            self.ng.stream = strm
            end = soff + min(subsz, totsz - soff)
            sbuf = sbuf.reshape((totsz, 1))
            ubtmp = dbuf.reshape((numrep, dbuf.size / numrep))
            sctx.push()
            strm.wait_for_event(evt)
            self.ng.sum(ubtmp, axis=0, out=sbuf[soff:end])
            sctx.pop()

        # REBROADCAST
        for dbuf, dctx in zip(ary.tlist, self.ctxs):
            for sbuf, sctx, soff, strm in zip(ary.tlist, self.ctxs,
                                              starts, strmlist):
                if sctx != dctx:
                    drv.memcpy_peer_async(dbuf.ptr + soff * dsz,
                                          sbuf.ptr + soff * dsz,
                                          min(subsz, totsz - soff) * dsz,
                                          dctx, sctx, strm)

        return ary
