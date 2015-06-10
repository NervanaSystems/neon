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
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)


class NoPar(object):

    def __init__(self):
        self.backend = None
        self.device_id = None

    def init_model(self, model, backend):
        backend.actual_batch_size = model.batch_size

    def associate(self, backend):
        backend.par = self
        self.backend = backend

    def distribute(self, batchdata, dtype):
        return self.backend.array(batchdata, dtype)

    def reduce_tensor(self, tensor):
        return tensor.asnumpyarray()

    def rank(self):
        return 0

    def is_distributed(self):
        return False


class BasePar(object):

    def __init__(self):
        self.backend = None
        self.device_id = None
        try:
            from mpi4py import MPI
            self.mpi = MPI
            self.comm = self.mpi.COMM_WORLD
            self.mpi_size = self.comm.size
            self.mpi_rank = self.comm.rank
        except ImportError:
            raise RuntimeError(
                "mpi4py not found, can't run in datapar or modelpar")

        try:
            # Determine local rank (assumes OpenMPI).
            self.mpi_local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
            self.mpi_local_size = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
        except:
            raise RuntimeError(
                "OpenMPI variable OMPI_COMM_WORLD_LOCAL_RANK or "
                "OMPI_COMM_WORLD_LOCAL_SIZE not found.\n"
                "Are you using: mpirun -n <#procs> neon <example.yaml>?")
        self.device_id = self.mpi_local_rank

    def init_model(self, model, backend):
        # save the original batch_size value that is specified in
        # the configuration file
        backend.actual_batch_size = model.batch_size

    def associate(self, backend):
        backend.par = self
        self.backend = backend

    def distribute(self, batchdata):
        raise NotImplementedError()

    def reduce_tensor(self, tensor):
        raise NotImplementedError()

    def distributable(self, layer):
        if hasattr(layer, 'distributable'):
            return layer.distributable
        return False

    def rank(self):
        return self.mpi_rank

    def is_distributed(self):
        return True


class ModelPar(BasePar):

    class Config(object):
        pass

    def __init__(self):
        super(ModelPar, self).__init__()
        if self.mpi_rank == 0:
            logger.info('Model-parallel mode. Number of nodes = %d.',
                        self.mpi_size)

    def init_model(self, model, backend):
        super(ModelPar, self).init_model(model, backend)
        for layer in model.layers:
            if not self.distributable(layer):
                continue
            assert hasattr(layer, 'nin')
            assert not hasattr(layer, 'parconf')
            conf = ModelPar.Config()
            nout = layer.nout
            realnin = layer.nin
            nin = realnin // self.mpi_size
            conf.start = self.mpi_rank * nin
            if self.mpi_rank == (self.mpi_size - 1):
                # If the weights cannot be evenly partitioned, let the last
                # MPI node handle the extra weights.
                conf.end = realnin
            else:
                conf.end = conf.start + nin
            bs = model.batch_size
            bufshape = (layer.nout, bs)
            conf.fpropbuf = np.empty(bufshape, dtype=np.float32)
            bufshape = (layer.nin, bs)
            conf.bpropbuf = np.empty(bufshape, dtype=np.float32)
            conf.rcount = np.empty(self.mpi_size, dtype=np.int32)
            conf.rcount.fill(nin)
            conf.scount = conf.end - conf.start
            conf.rcount[-1] = realnin - nin * (self.mpi_size - 1)
            conf.displ = np.arange(0, realnin - nin + 1, nin)
            conf.scount *= bs
            conf.rcount *= bs
            conf.displ *= bs
            layer.weight_shape = (nout, conf.end - conf.start)
            layer.parconf = conf

    def associate(self, backend):
        super(ModelPar, self).associate(backend)
        self.orig_fprop_fc = backend.fprop_fc
        self.orig_bprop_fc = backend.bprop_fc
        self.orig_update_fc = backend.update_fc
        backend.fprop_fc = self.fprop_fc
        backend.bprop_fc = self.bprop_fc
        backend.update_fc = self.update_fc

    def distribute(self, batchdata, dtype):
        return self.backend.array(batchdata, dtype)

    def reduce_tensor(self, tensor):
        return tensor.asnumpyarray()

    def fprop_fc(self, out, inputs, weights, layer):
        conf = layer.parconf
        self.orig_fprop_fc(out, inputs[conf.start:conf.end], weights)
        sendbuf = [out.asnumpyarray(), self.mpi.FLOAT]
        recvbuf = [conf.fpropbuf, self.mpi.FLOAT]
        self.comm.Reduce(sendbuf, recvbuf, op=self.mpi.SUM)
        self.comm.Bcast(buf=[conf.fpropbuf, self.mpi.FLOAT])
        out.copy_from(conf.fpropbuf)

    def bprop_fc(self, out, weights, deltas, layer):
        conf = layer.parconf
        self.orig_bprop_fc(out[conf.start:conf.end], weights, deltas)
        outbuf = out.asnumpyarray()[conf.start:conf.end]
        sendbuf = [outbuf, conf.scount, self.mpi.FLOAT]
        recvbuf = [conf.bpropbuf, conf.rcount,
                   conf.displ, self.mpi.FLOAT]
        self.comm.Allgatherv(sendbuf, recvbuf)
        out.copy_from(conf.bpropbuf)

    def update_fc(self, out, inputs, deltas, layer):
        conf = layer.parconf
        self.orig_update_fc(out, inputs[conf.start:conf.end], deltas)


class DataPar(BasePar):

    class Config(object):
        pass

    def __init__(self):
        super(DataPar, self).__init__()
        if self.mpi_rank == 0:
            logger.info('Data-parallel mode. Number of nodes = %d.',
                        self.mpi_size)
        self.reducebuf = np.empty((1, 1), dtype=np.float32)

    def init_model(self, model, backend):
        super(DataPar, self).init_model(model, backend)
        self.batch_size = backend.actual_batch_size // self.mpi_size
        self.start = self.mpi_rank * self.batch_size
        if self.mpi_rank == (self.mpi_size - 1):
            self.batch_size = backend.actual_batch_size - self.start
        self.end = self.start + self.batch_size
        model.batch_size = self.batch_size

        for layer in model.layers:
            if not self.distributable(layer):
                continue
            assert hasattr(layer, 'nin')
            assert not hasattr(layer, 'parconf')
            conf = DataPar.Config()
            conf.updatebuf = np.empty(layer.weight_shape, dtype=np.float32)
            layer.parconf = conf

    def associate(self, backend):
        super(DataPar, self).associate(backend)
        self.orig_update_fc = backend.update_fc
        self.orig_update_conv = backend.update_conv
        backend.update_fc = self.update_fc
        backend.update_conv = self.update_conv

    def distribute(self, batchdata, dtype):
        return self.backend.array(batchdata[:, self.start:self.end], dtype)

    def reduce_tensor(self, tensor):
        self.comm.Reduce([tensor.asnumpyarray(), self.mpi.FLOAT],
                         [self.reducebuf, self.mpi.FLOAT], op=self.mpi.SUM)
        if self.mpi_rank == 0:
            return self.reducebuf / self.mpi_size
        return 0

    def update(self, out, conf):
        # NOTE: To make this faster, compute the weight updates
        # asynchronously. There is no need to wait for completion
        # until the updates are to be applied to the weights (the
        # weights are updated after the gradients are propagated
        # all the way back).
        sendbuf = [out.asnumpyarray(), self.mpi.FLOAT]
        recvbuf = [conf.updatebuf, self.mpi.FLOAT]
        self.comm.Reduce(sendbuf, recvbuf, op=self.mpi.SUM)
        self.comm.Bcast(buf=[conf.updatebuf, self.mpi.FLOAT])
        out.copy_from(conf.updatebuf)

    def update_fc(self, out, inputs, deltas, layer):
        self.orig_update_fc(out, inputs, deltas)
        self.update(out, layer.parconf)

    def update_conv(self, out, inputs, weights, deltas, ofmshape, ofmsize,
                    ofmlocs, ifmshape, links, nifm, padding, stride,
                    ngroups, fwidth, updatebuf, local=False, layer=None):
        self.orig_update_conv(out, inputs, weights, deltas, ofmshape, ofmsize,
                              ofmlocs, ifmshape, links, nifm, padding, stride,
                              ngroups, fwidth, updatebuf, local)
        self.update(out, layer.parconf)
