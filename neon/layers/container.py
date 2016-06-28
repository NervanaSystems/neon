# ----------------------------------------------------------------------------
# Copyright 2014-2016 Nervana Systems Inc.
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

from builtins import str, zip, range
import numpy as np
import itertools as itt
from operator import add

from neon import NervanaObject
from neon.layers.layer import Layer, BranchNode, Dropout, DataTransform, LookupTable
from neon.layers.recurrent import Recurrent, get_steps
from neon.util.persist import load_class
from functools import reduce


def flatten(item):
    if hasattr(item, '__iter__'):
        for i in iter(item):
            for j in flatten(i):
                yield j
    else:
        yield item


class DeltasTree(NervanaObject):
    """
    Data structure for maintaining nested global delta buffers
    """
    def __init__(self, parent=None):
        self.parent = None
        self.child = None
        self.buffers = [None]*2
        self.max_shape = 0
        if parent:
            assert type(parent) is DeltasTree
            self.parent = parent

    def decend(self):
        if self.child is None:
            self.child = DeltasTree()
        return self.child

    def ascend(self):
        return self.parent

    def proc_layer(self, layer):
        in_size = layer.be.shared_iobuf_size(layer.in_shape,
                                             layer.parallelism)
        if in_size > self.max_shape:
            self.max_shape = in_size

    def allocate_buffers(self):
        if self.child:
            self.child.allocate_buffers()

        for ind in range(len(self.buffers)):
            if self.buffers[ind] is None:
                if self.max_shape > 0:
                    self.buffers[ind] = self.be.iobuf(self.max_shape,
                                                      persist_values=False,
                                                      parallelism="Data")


class LayerContainer(Layer):
    """
    Layer containers are a generic class that are used to encapsulate groups of layers and
    provide methods for propagating through the constituent layers, allocating memory.
    """
    @property
    def layers_to_optimize(self):
        lto = []
        for l in self.layers:
            if isinstance(l, LayerContainer):
                lto += l.layers_to_optimize
            elif l.has_params:
                if hasattr(l, 'init') and l.init.name == "Identity":
                    continue
                lto.append(l)
        return lto

    @property
    def nest_deltas(self):
        return False

    def nested_str(self, level=0):
        """
        Utility function for displaying layer info with a given indentation level.

        Arguments:
            level (int, optional): indentation level

        Returns:
            str: layer info at the given indentation level
        """
        padstr = '\n' + '  ' * level
        ss = '  ' * level + self.classnm + padstr
        ss += padstr.join([l.nested_str(level + 1) for l in self.layers])
        return ss

    @classmethod
    def gen_class(cls, pdict):
        layers = []

        for layer in pdict['layers']:
            typ = layer['type']
            ccls = load_class(typ)
            layers.append(ccls.gen_class(layer['config']))

        # the 'layers' key  is special in that the layer
        # parameters are in there and need to be saved the
        # whole pdict['layers'] element can not be replaced
        # with the just the layer objects like elsewhere
        lsave = pdict.pop('layers')
        new_cls = cls(layers=layers, **pdict)
        pdict['layers'] = lsave
        return new_cls

    def get_description(self, get_weights=False, keep_states=False):
        """
        Get layer parameters. All parameters are needed for optimization, but
        only weights are serialized.

        Arguments:
            get_weights (bool, optional): Control whether all parameters are returned or
                                          just weights for serialization.
            keep_states (bool, optional): Control whether all parameters are returned
                                          or just weights for serialization.
        """
        desc = super(LayerContainer, self).get_description(skip=['layers'])
        desc['container'] = True
        desc['config']['layers'] = []
        for layer in self.layers:
            desc['config']['layers'].append(layer.get_description(get_weights=get_weights,
                                                                  keep_states=keep_states))
        self._desc = desc
        return desc

    def load_weights(self, pdict, load_states=True):
        """
        Load weights.

        Arguments:
            pdict:
            load_states:  (Default value = True)

        Returns:

        """
        assert len(pdict['config']['layers']) == len(self.layers)
        for branch, bdict in zip(self.layers, pdict['config']['layers']):
            branch.load_weights(bdict, load_states=load_states)

    def revert_tensors(self):
        for tensor in itt.chain.from_iterable([l.revert_list for l in self.layers]):
            self.be.revert_tensor(tensor)

    def propagate_parallelism(self, p):
        for l in self.layers:
            if isinstance(l, LayerContainer):
                l.parallelism = p
                l.propagate_parallelism(p)
                t = l.get_terminal()
                p = t[0].parallelism if isinstance(t, list) else t.parallelism
            else:
                l.parallelism = p if l.parallelism == "Unknown" else l.parallelism
                p = l.parallelism

    def set_batch_size(self, N):
        """
        Set minibatch size.

        Arguments:
            N (int): minibatch size
        """
        for l in self.layers:
            l.set_batch_size(N)

    def set_seq_len(self, S):
        """
        Set sequence length.

        Arguments:
            S (int): sequence length
        """
        for l in self.layers:
            l.set_seq_len(S)

    def set_deltas(self, global_deltas):
        """
        Set the layer deltas from the shared
        global deltas pool
        """
        for l in self.layers:
            l.set_deltas(global_deltas)

    def layers_fprop(self):
        """
        Generator to iterator over the layers in the same
        order as fprop
        """
        for layer in self.layers:
            yield layer
            if hasattr(layer, 'layers_fprop'):
                for layer2 in layer.layers_fprop():
                    yield layer2

    def layers_bprop(self):
        """
        Generator to iterator over the layers in the same
        order as bprop
        """
        for layer in reversed(self.layers):
            if hasattr(layer, 'layers_bprop'):
                for layer2 in layer.layers_bprop():
                    yield layer2
            yield layer


class Sequential(LayerContainer):
    """
    Layer container that encapsulates a simple linear pathway of layers.

    Arguments:
        layers (list): List of objects which can be either a list of layers
                       (including layer containers).
    """
    def __init__(self, layers, name=None):
        super(Sequential, self).__init__(name)

        self.layers = [l for l in flatten(layers)]
        self._layers = [x for x in self.layers if type(x) not in (BranchNode,)]
        root = self._layers[0]
        assert (root.owns_output or
                type(root) in [Dropout, DataTransform]), "Sequential root must own outputs"

    def configure(self, in_obj):
        """
        Must receive a list of shapes for configuration (one for each pathway)
        the shapes correspond to the layer_container attribute

        Arguments:
            in_obj: any object that has an out_shape (Layer) or shape (Tensor, dataset)
        """
       if in_obj:
            config_layers = self.layers
            in_obj = in_obj
        else:
            in_obj = self.layers[0]

            # Remove the initial branch nodes from the layers
            for l_idx, l in enumerate(self.layers):
                if type(l) in (BranchNode,):
                    continue
                else:
                    config_layers = self.layers[l_idx:]
                    break

        super(Sequential, self).configure(in_obj)
        prev_layer = None
        for l in config_layers:
            in_obj = l.configure(in_obj)
            if prev_layer is not None:
                prev_layer.set_next(l)
            prev_layer = l
        self.parallelism = in_obj.parallelism
        self.out_shape = in_obj.out_shape
        return self

    def allocate(self, shared_outputs=None):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
        """
        # get the layers that own their outputs
        alloc_layers = [l for l in self.layers if l.owns_output]
        alloc_layers[-1].allocate(shared_outputs)
        for l in self.layers:
            l.allocate()

    def allocate_deltas(self, global_deltas=None):
        if global_deltas is None:
            self.global_deltas = DeltasTree()

            st_ind = 0 if getattr(self.layers[0], 'nest_deltas', False) else 1
            for layer in self.layers[st_ind:]:
                layer.allocate_deltas(self.global_deltas)

            self.global_deltas.allocate_buffers()

        self.set_deltas(self.global_deltas)

    def fprop(self, inputs, inference=False, beta=0.0):
        """
        TODO:  Handle final layers that don't own their own outputs (bias, activation)

        Arguments:
            inputs:
            inference:  (Default value = False)
            beta:  (Default value = 0.0)

        Returns:

        """
        x = inputs

        for l in self.layers:
            altered_tensor = l.be.distribute_data(x, l.parallelism)
            l.revert_list = [altered_tensor] if altered_tensor else []

            if l is self.layers[-1] and beta != 0:
                x = l.fprop(x, inference=inference, beta=beta)
            else:
                x = l.fprop(x, inference=inference)

        if inference:
            self.revert_tensors()

        return x

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        for l in reversed(self._layers):
            altered_tensor = l.be.distribute_data(error, l.parallelism)
            if altered_tensor:
                l.revert_list.append(altered_tensor)
            
            if type(l.prev_layer) is BranchNode or l is self._layers[0]:
                error = l.bprop(error, alpha, beta)
            else:
                error = l.bprop(error)

            for tensor in l.revert_list:
                self.be.revert_tensor(tensor)
        return self._layers[0].deltas

    def get_terminal(self):
        """
        Used for recursively getting final nodes from layer containers.
        """
        terminal = self.layers[-1].get_terminal()
        return terminal


class Tree(LayerContainer):
    """
    Layer container that encapsulates a simple linear pathway of layers.

    Arguments:
        layers (list): List of Sequential containers corresponding to the branches of the Tree.
                       The branches must be provided with main trunk first, and then the auxiliary
                       branches in the order the branch nodes are encountered
        name (string, optional): Name for the container
        alphas (list(float), optional): list of weighting factors to apply to each branch for
                                        backpropagating error.
    """

    def __init__(self, layers, name=None, alphas=None):
        super(Tree, self).__init__(name=name)
        self.layers = []
        for l in layers:
            if isinstance(l, Sequential):
                self.layers.append(l)
            elif isinstance(l, list):
                self.layers.append(Sequential(l))
            elif isinstance(l, Layer):
                self.layers.append(Sequential([l]))
            else:
                ValueError("Incompatible element for Tree container")

        self.alphas = [1.0 for _ in self.layers] if alphas is None else alphas

        # alphas and betas are used for back propagation
        # We want to ensure that the branches are ordered according to the origin of their roots
        # then the betas will be 0 for the last appearance of the root, and 1 for the rest,
        # but the trunk will always be 1 (since it contains all of the branch nodes)
        self.betas = []
        next_root = None
        for l in reversed(self.layers):
            root = l.layers[0]
            beta = 1.0 if (root is next_root or type(root) is not BranchNode) else 0.0
            next_root = root
            self.betas.append(beta)
        self.betas.reverse()

    def nested_str(self, level=0):
        """
        Utility function for displaying layer info with a given indentation level.

        Arguments:
            level (int, optional): indentation level

        Returns:
            str: layer info at the given indentation level
        """
        ss = self.classnm + '\n'
        ss += '\n'.join([l.nested_str(level + 1) for l in self.layers])
        return ss

    def configure(self, in_obj):
        """
        Set shape based parameters of this layer given an input tuple, int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer, Tensor or dataset): object that provides shape
                                                           information for layer

        Returns:
            (tuple): shape of output data
        """
        super(Tree, self).configure(in_obj)
        self.layers[0].configure(in_obj)
        for l in self.layers[1:]:
            l.configure(None)
        self.out_shape = [l.out_shape for l in self.layers]
        return self

    def allocate(self, shared_outputs=None):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
        """
        for l in self.layers:
            l.allocate()
        self.outputs = [l.outputs for l in self.layers]

    def allocate_deltas(self, global_deltas=None):
        for l in reversed(self.layers):
            l.allocate_deltas(global_deltas)

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data

        Returns:
            Tensor: output data
        """
        x = self.layers[0].fprop(inputs, inference)
        out = [x] + [l.fprop(None, inference=inference) for l in self.layers[1:]]
        return out

    def bprop(self, error):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        for l, e, a, b in reversed(list(zip(self.layers, error, self.alphas, self.betas))):
            l.bprop(e, alpha=a, beta=b)

    def get_terminal(self):
        """
        Used for recursively getting final nodes from layer containers.
        """
        return [l.get_terminal() for l in self.layers]


class SingleOutputTree(Tree):
    """
    Subclass of the Tree container which returns only
    the output of the main branch (branch index 0) during
    inference.
    """
    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data

        Returns:
            Tensor: output data
        """
        x = self.layers[0].fprop(inputs, inference)
        if inference:
            return x
        else:
            out = [x] + [l.fprop(None) for l in self.layers[1:]]
            return out


class Broadcast(LayerContainer):
    """
    Parent class for MergeSum and MergeBroadcast.
    """
    def __init__(self, layers, name=None):
        super(Broadcast, self).__init__(name)
        # Input list of layers converts:
        #   lists to Sequential container
        #   singleton layers to Sequential containers of 1
        #   leaves Sequentials alone
        self.layers = []
        for l in layers:
            if isinstance(l, Sequential):
                self.layers.append(l)
            elif isinstance(l, list):
                self.layers.append(Sequential(l))
            elif isinstance(l, Layer):
                self.layers.append(Sequential([l]))
            else:
                ValueError("Incompatible element for " + self.__class__.__name__ + " Layer")
        self.owns_output = True
        self.outputs = None

    @property
    def nest_deltas(self):
        return True

    def __str__(self):
        ss = '\n\t'.join([str(l) for l in self.layers])
        ss = '\t' + self.classnm + '\n\t' + ss
        return ss

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer

        Arguments:
            in_obj (int, tuple, Layer or Tensor or dataset): object that provides shape
                                                             information for layer

        Returns:
            (tuple): shape of output data
        """
        super(Broadcast, self).configure(in_obj)

        # Receiving from single source -- distribute to branches
        for l in self.layers:
            l.configure(in_obj)
        self._configure_merge()
        return self

    def allocate_deltas(self, global_deltas):
        nested_deltas = global_deltas.decend()
        for layer in self.layers:
            layer.layers[0].allocate_deltas(global_deltas)
            for sublayer in layer.layers[1:]:
                sublayer.allocate_deltas(nested_deltas)

    def set_deltas(self, delta_buffers):
        """
        Use pre-allocated (by layer containers) list of buffers for backpropagated error.
        Only set deltas for layers that own their own deltas
        Only allocate space if layer owns its own deltas (e.g., bias and activation work in-place,
        so do not own their deltas).

        Arguments:
            delta_buffers (DeltasTree): list of pre-allocated tensors (provided by layer container)
        """
        bottom_buffer = delta_buffers.buffers[0]

        nested_deltas = delta_buffers.decend()
        assert nested_deltas is not None
        for l in self.layers:
            l.layers[0].set_deltas(delta_buffers)
            delta_buffers.buffers.reverse()  # undo that last reverse
            for sublayer in l.layers[1:]:
                sublayer.set_deltas(nested_deltas)

        # Special case if originating from a branch node
        if type(self.prev_layer) is BranchNode:
            self.deltas = self.be.iobuf(self.in_shape, shared=self.prev_layer.deltas,
                                        parallelism=self.parallelism)
        else:
            self.deltas = self.be.iobuf(self.in_shape, shared=bottom_buffer,
                                        parallelism=self.parallelism)
            delta_buffers.buffers.reverse()

    def get_terminal(self):
        """
        Used for recursively getting final nodes from layer containers.
        """
        terminals = [l.get_terminal() for l in self.layers]
        return terminals


class MergeSum(Broadcast):
    """
    """

    def allocate(self, shared_outputs=None):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
        """
        if self.outputs is None:
            self.outputs = self.be.iobuf(self.out_shape, shared=shared_outputs,
                                         parallelism=self.parallelism)
        for l in self.layers:
            l.allocate(shared_outputs=self.outputs)

    def _configure_merge(self):
        """
        Helper function for configuring output shape
        """
        out_shapes = [l.out_shape for l in self.layers]
        self.out_shape = out_shapes[0]

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data

        Returns:
            Tensor: output data
        """
        for l in self.layers:
            beta = 0 if l is self.layers[0] else 1
            l.fprop(inputs, inference, beta=beta)
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        for l in reversed(self.layers):
            b = beta if l is self.layers[-1] else 1
            l.bprop(error, alpha=alpha, beta=b)
        return self.deltas


class MergeBroadcast(Broadcast):
    """
    Branches a single incoming layer or object (broadcast) into multiple output paths that are
    then combined again (merged). This container supports several options for concatenating the
    paths ("recurrent", "depth", and "stack").

    "recurrent" is used when merging two recurrent output streams.

    "depth" concatenates activations that have a notion of spatial dimension. Multiple
    activations can be concatenated along the feature map dimension, but the feature map
    shapes have to be the same.

    "stack" ignores the feature map shape and simply stacks the non-batch dimensions
    atop each other. Used to concatenate the output of fully connected layers with each
    other, and fully connected layers with convolutional layers.

    For example, suppose we are merging a conv layer with output shape (10, 5, 5)
    and a fully connected layer with 100 output nodes. Using 'depth' is not allowable.
    By using 'stack', the (10, 5, 5) output of the conv layer would just be interpreted as
    250 output nodes that are stacked on top of the 100 nodes from the fully connected
    layer to get a total merged output of 350 nodes.

    Arguments:
        layers (list(list(Layer), LayerContainer): list of either layer lists,
                                                   or layer containers.  Elements that are
                                                   lists will be wrapped in Sequential
                                                   containers
        merge (string): the merging method. Must be 'recurrent', 'depth', or 'stack'
        alphas (list(float), optional):  list of alpha values by which to weight the
                                         backpropagated errors
        name (str): Container name.  Defaults to "MergeBroadcast"
    """
    def __init__(self, layers, merge, alphas=None, name=None):
        super(MergeBroadcast, self).__init__(layers, name)

        self.betas = [1.0 for _ in self.layers]
        self.betas[-1] = 0.0
        self.alphas = [1.0 for _ in self.layers] if alphas is None else alphas

        self.merge = merge  # How this MergeBroadcast gets merged
        assert self.merge in ("recurrent", "depth", "stack")
        self.error_views = None

    def get_partitions(self, x, slices):
        """
        Given a partitioning, slices, of an activation buffer, x, determine which axis to slice
        along depending on whether x is a sequential tensor or not.

        Arguments:
            x:
            slices:

        Returns:

        """
        if x.shape[-1] != self.be.bsz:  # This is the sequential case
            return [x[:, sl] for sl in slices]
        else:
            return [x[sl] for sl in slices]

    def allocate(self, shared_outputs=None):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
        """
        if self.outputs is None:
            self.outputs = self.be.iobuf(self.out_shape, shared=shared_outputs,
                                         parallelism=self.parallelism)
        self.output_views = self.get_partitions(self.outputs, self.slices)
        for l, out_view in zip(self.layers, self.output_views):
            l.allocate(shared_outputs=out_view)

    def _configure_merge(self):
        """
        Helper function for configuring shapes depending on the merge concatenation type
        """
        in_shapes = [l.out_shape for l in self.layers]
        # Figure out how to merge
        if self.merge == "recurrent":
            catdims = [xs[1] for xs in in_shapes]
            self.out_shape = (in_shapes[0][0], sum(catdims))
            stride_size = self.be.bsz
        elif self.merge == "depth":
            catdims = [xs[0] for xs in in_shapes]
            self.out_shape = (sum(catdims),) + in_shapes[0][1:]
            stride_size = np.prod(in_shapes[0][1:])
        elif self.merge == "stack":
            catdims = [xs if isinstance(xs, int) else np.prod(xs) for xs in in_shapes]
            self.out_shape = sum(catdims)
            stride_size = 1
        end_idx = [idx * stride_size for idx in np.cumsum(catdims)]
        start_idx = [0] + end_idx[:-1]
        self.slices = [slice(s, e) for s, e in zip(start_idx, end_idx)]

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data

        Returns:
            Tensor: output data
        """
        for l in self.layers:
            l.fprop(inputs, inference)
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        self.betas[-1] = beta
        if self.error_views is None:
            self.error_views = self.get_partitions(error, self.slices)
        for l, e, a, b in reversed(list(zip(self.layers, self.error_views, self.alphas,
                                            self.betas))):
            l.bprop(e, alpha=a * alpha, beta=b)
        return self.deltas


class MergeMultistream(MergeBroadcast):
    """
    Merging multiple input sources via concatenation.  This container is similar to MergeBroadcast
    except that it receives different streams of input directly from a dataset.
    """
    def __init__(self, layers, merge, name=None):
        super(MergeMultistream, self).__init__(layers, merge=merge, name=name)

    @property
    def nest_deltas(self):
        return False

    def configure(self, in_obj):
        """
        Must receive a list of shapes for configuration (one for each pathway)
        the shapes correspond to the layer_container attribute

        Arguments:
            in_obj (list(Tensor)): list of Data tensors provided to each sequential container
        """
        self.prev_layer = None
        if not isinstance(in_obj, list):
            assert hasattr(in_obj, 'shape') and isinstance(in_obj.shape, list)
            in_obj = in_obj.shape
        assert isinstance(in_obj, list), "Multistream inputs must be interpretable as shapes"
        for inp, l in zip(in_obj, self.layers):
            l.configure(inp)
        self._configure_merge()
        return self

    def set_deltas(self, delta_buffers):
        """
        Use pre-allocated (by layer containers) list of buffers for backpropagated error.
        Only set deltas for layers that own their own deltas
        Only allocate space if layer owns its own deltas (e.g., bias and activation work in-place,
        so do not own their deltas).

        Arguments:
            delta_buffers (list): list of pre-allocated tensors (provided by layer container)
        """
        # delta_buffers ignored here, will generate
        # new delta buffers for each sequential container
        for l in self.layers:
            l.allocate_deltas()

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data

        Returns:
            Tensor: output data
        """
        for l, inp in zip(self.layers, inputs):
            l.fprop(inp, inference)
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if self.error_views is None:
            self.error_views = self.get_partitions(error, self.slices)
        for l, e in zip(self.layers, self.error_views):
            l.bprop(e)


class RoiPooling(Sequential):
    """
    It uses max pooling to convert the features inside any ROI into a small
    feature map with a fixed spatial extend of H x W, where H and W are layer
    parameters indepdendent of any particular ROI.
    Each ROI is defined as a 4-tuple as (xmin, ymin, xmax, ymax)

    ROIPooling is applied independently to each feature map channel, as in standard
    max pooling.

    It is constructed as a layer container, in order to interface with dataset
    directly. And it will process the image from the dataset through the contained
    layers (usually ImageNet CNN layers), and combine the ROIs from the dataset
    with the feature maps output from CNN layers.

    The RoiPooling container processes images with preset batch size. While after
    the ROI pooling, the minibatch is extended into batch_size * rois_per_img
    examples in each minibatch.
    The output shape (out_shape) is a tuple - (batch_size, rois_per_img), then
    the following layers will allocate buffers accordingly.
    """

    def __init__(self, layers, bprop_enabled=False, HW=(7, 7),
                 spatial_scale=0.0625, name=None):
        if layers:
            super(RoiPooling, self).__init__(layers, name=name)
        self.HW = HW
        self.roi_H, self.roi_W = self.HW
        self.spatial_scale = spatial_scale  # 0.0625 is 1/16
        # it has its own output buffer besides being a container
        self.owns_output = True
        self.owns_delta = True
        self.img = None
        self.rois = None
        self.rois_per_img = 64
        self.rois_per_batch = self.be.bsz * 64
        self.bprop_enabled = bprop_enabled

    def nested_str(self, level=0):
        """
        Utility function for displaying layer info with a given indentation level.

        Arguments:
            level (int, optional): indentation level

        Returns:
            str: layer info at the given indentation level
        """
        ss = self.__class__.__name__ + '\n'
        return ss

    def configure(self, in_obj):
        """
        Must receive a list of shapes for configurations
        Need both the layer container and roi dataset to configure shapes
        'in_obj' will include be [image_shape, roi_shape] (e.g [(3, 600, 1000), 5])

        Arguments:
            in_obj:

        Returns:

        """
        # configure to get the shape of feature map
        self.prev_layer = None

        if not isinstance(in_obj, list):
            assert hasattr(in_obj, 'shape') and isinstance(in_obj.shape, list)
            # make sure the in_obj has information on rois_per_img,
            # if it is a dataset
            assert hasattr(in_obj, 'rois_per_img')
            self.rois_per_img = in_obj.rois_per_img
            assert hasattr(in_obj, 'rois_per_batch')
            self.rois_per_batch = in_obj.rois_per_batch

            in_obj = in_obj.shape

        assert isinstance(
            in_obj, list), "ROI pool layer must have interpretable input shapes"

        # configure all the image network layers, so self.layers has feature map shape
        # using which to get output shape

        in_obj_img = in_obj[0]
        self.in_shape_img = in_obj_img  # the previous sequential in shape

        if self.layers:
            for l in self.layers:
                in_obj_img = l.configure(in_obj_img)
            self.in_shape = in_obj_img.out_shape  # ROI layer in shape
            (self.fm_channel, self.fm_height, self.fm_width) = self.in_shape
            self.fm_reshape_shape = (
                self.fm_channel, self.fm_height * self.fm_width, self.be.bsz)
            self.error_in_reshape = (self.fm_channel, -1)

        # make the out_shape as a tuple, as if the roi_per_image a
        # time_step dimension
        self.out_shape = (self.fm_channel * self.roi_H * self.roi_W, self.rois_per_img)
        return self

    def allocate(self, shared_outputs=None):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
        """
        super(RoiPooling, self).allocate(shared_outputs)
        self.owns_output = True
        self.error = self.be.iobuf(self.in_shape)
        self.outputs = self.be.iobuf(self.out_shape, shared=shared_outputs)
        self.max_idx = self.be.iobuf(self.out_shape, dtype=np.int32)

    def allocate_deltas(self, global_deltas=None):
        if global_deltas is None:
            self.global_deltas = DeltasTree()

            st_ind = 0 if getattr(self.layers[0], 'nest_deltas', False) else 1
            for layer in self.layers[st_ind:]:
                layer.allocate_deltas(self.global_deltas)

            self.global_deltas.allocate_buffers()
        super(RoiPooling, self).set_deltas(self.global_deltas)

    def set_deltas(self, delta_buffers):
        """
        Use pre-allocated (by layer containers) list of buffers for backpropagated error.
        Only set deltas for layers that own their own deltas
        Only allocate space if layer owns its own deltas (e.g., bias and activation work in-place,
        so do not own their deltas).

        Arguments:
            delta_buffers (list): list of pre-allocated tensors (provided by layer container)
        """
        self.allocate_deltas()

    def init_buffers(self, inputs):
        """
        Initialize buffers for images and ROIs

        Arguments:
            inputs:

        Returns:

        """
        assert len(inputs) == 2, "inputs must contain both images and ROIs"
        if self.img is None or self.img is not inputs[0]:
            self.img = inputs[0]
            self.rois = inputs[1]
            assert self.rois.shape[1] == 5, "ROI entry must be 5-value tuple"

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data

        Returns:
            Tensor: output data
        """
        self.init_buffers(inputs)

        self.outputs.fill(0)
        self.max_idx.fill(0)

        # fprop the input images
        self.fm = super(RoiPooling, self).fprop(self.img, inference)

        # fprop through the roipooling layer
        self.be.roipooling_fprop(self.fm, self.rois, self.outputs, self.max_idx,
                                 self.rois_per_batch, self.fm_channel, self.fm_height,
                                 self.fm_width, self.roi_H, self.roi_W, self.spatial_scale)

        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """

        self.error.fill(0)

        if self.bprop_enabled:
            # # bprop through the roipooling layer
            self.be.roipooling_bprop(error, self.rois, self.error, self.max_idx,
                                     self.rois_per_batch, self.fm_channel, self.fm_height,
                                     self.fm_width, self.roi_H, self.roi_W, self.spatial_scale)

        # bprop back through the imagenet layer container
        self.deltas = super(RoiPooling, self).bprop(self.error, alpha, beta)

    def get_terminal(self):
        """
        Used for recursively getting final nodes from layer containers.
        """
        terminals = [l.get_terminal() for l in self.layers]
        return terminals


class Encoder(Sequential):
    """
    Encoder stack for the Seq2Seq container. Acts like a sequential
    except for bprop which are connected as specified to Decoder recurrent layers
    """
    def __init__(self, layers, name=None):
        super(Encoder, self).__init__(layers, name)
        # list of recurrent layers only:
        self._recurrent = [l for l in self.layers if isinstance(l, Recurrent)]
        self.connections = None
        self.error_buf = None
        self.error_slices = None

    def allocate_deltas(self, global_deltas=None):
        super(Encoder, self).allocate_deltas(global_deltas=global_deltas)

        self.error_buf = self.be.iobuf(self.out_shape)
        self.error_slices = get_steps(self.error_buf, self.out_shape)

    def set_connections(self, decoder_cons):
        """
        Based on decoder connections, create the list of which layers encoder are
        connected to.
        """
        cons = []
        for ii in range(len(self._recurrent)):
            l_list = [i_dec for i_dec, i_enc in enumerate(decoder_cons) if i_enc == ii]
            cons.append(l_list)
        self.connections = cons

    def get_final_states(self, decoder_cons):
        """
        Based on decoder connections, prepare the list of final states for decoder
        """
        final_states = [self._recurrent[ii].final_state()
                        if ii is not None else None
                        for ii in decoder_cons]

        return final_states

    def bprop(self, hidden_error_list, inference=False, alpha=1.0, beta=0.0):
        """
        Arguments:
            hidden_error_list: Decoder container bprop output. List of errors
                               associated with decoder recurrent layers.
        """
        i_enc = len(self._recurrent) - 1  # index into recurrent layers, in reverse order

        # initialize error to zeros (shape of last encoder layer output)
        error = self.error_buf
        error.fill(0)

        # bprop through layers, setting up connections from decoder layers for recurrent layers
        for l in reversed(self._layers):
            altered_tensor = l.be.distribute_data(error, l.parallelism)
            if altered_tensor:
                l.revert_list.append(altered_tensor)

            # add the hidden error by the hidden error list
            if isinstance(l, Recurrent):
                for i_dec in self.connections[i_enc]:
                    self.error_slices[-1][:] = self.error_slices[-1] + hidden_error_list[i_dec]
                i_enc -= 1

            # normal bprop through the layers
            if type(l.prev_layer) is BranchNode or l is self._layers[0]:
                error = l.bprop(error, alpha, beta)
            else:
                error = l.bprop(error)

            for tensor in l.revert_list:
                self.be.revert_tensor(tensor)


class Decoder(Sequential):
    """
    Decoder stack for the Seq2Seq container. Acts like a sequential
    except for fprop which takes the additional init_state_list, and bprop
    which takes additional hidden_delta
    """
    def __init__(self, layers, name=None):
        super(Decoder, self).__init__(layers, name)
        # list of recurrent layers only:
        self._recurrent = [l for l in self.layers if isinstance(l, Recurrent)]
        self.connections = None
        self.full_steps = None

    def fprop(self, x, inference=False, init_state_list=None):

        if init_state_list is None:
            init_state_list = [None for _ in range(len(self._recurrent))]

        ii = 0  # index into init_state_list (decoder recurrent layer number)
        for l in self.layers:
            altered_tensor = l.be.distribute_data(x, l.parallelism)
            l.revert_list = [altered_tensor] if altered_tensor else []

            # special fprop for recurrent layers with init state
            if isinstance(l, Recurrent):
                x = l.fprop(x, inference=inference, init_state=init_state_list[ii])
                ii = ii + 1
            else:
                x = l.fprop(x, inference=inference)

        return x

    def set_connections(self, decoder_cons):
        self.connections = decoder_cons

    def bprop(self, error, inference=False, alpha=1.0, beta=0.0):
        """
        bprop through layers, saving hidden_error for Recurrent layers
        """
        hidden_error_list = []
        for l in reversed(self.layers):
            altered_tensor = l.be.distribute_data(error, l.parallelism)
            if altered_tensor:
                l.revert_list.append(altered_tensor)

            error = l.bprop(error)
            if isinstance(l, Recurrent):
                hidden_error_list.append(l.get_final_hidden_error())

            for tensor in l.revert_list:
                self.be.revert_tensor(tensor)

        # return hidden error in order of decoder layers
        # (to match decoder_connections)
        hidden_error_list.reverse()

        return hidden_error_list

    def switch_mode(self, inference, conditional):
        """
        Dynamically grow or shrink the number of time steps to perform
        single time step fprop during inference.
        """
        # set up parameters
        hasLUT = isinstance(self.layers[0], LookupTable)
        # sequence length is different dimension depending on whether there is LUT
        cur_steps = self.in_shape[0] if hasLUT else self.in_shape[1]
        if not (inference and conditional):
            old_size = cur_steps
            # assumes encoder and decoder have the same sequence length
            new_size = self.full_steps
        else:
            old_size = cur_steps
            new_size = 1
        # resize buffers
        if old_size != new_size:
            if hasLUT:
                in_obj = (new_size, 1)
            else:
                in_obj = (self.out_shape[0], new_size)
            self.configure(in_obj=in_obj)
            # set layer outputs to None so they get reallocated
            for l in self.layers:
                if l.owns_output:
                    l.outputs = None
            self.allocate(shared_outputs=None)  # re-allocate deltas, but not weights
            for l in self.layers:
                l.name += "'"


class Seq2Seq(LayerContainer):
    """
    Layer container that encapsulates encoder decoder pathways
    used for sequence to sequence models.

    Arguments:
        layers (list): Length two list specifying the encoder and decoder.
                       The encoder must be provided as the first list element.
                       List elements may be an Encoder and a Decoder container, or,
                       similar to Tree and Broadcast containers, encoder (decoder) can be
                       specified as a list of layers or a single layer, which are
                       converted to Encoder and Decoder containers.
        decoder_connections (list of ints): for every recurrent decoder layer, specifies the
                                         corresponding encoder layer index (recurrent layers only)
                                         to get initial state from. The format will be, e.g.
                                         [0, 1, None].
                                         If not given, the container will try to make a
                                         one-to-one connections, which assumes an equal number
                                         of encoder and decoder recurrent layers.
        conditional (boolean): if True, decoder uses output at previous time step as input
    """
    def __init__(self, layers, decoder_connections=None, conditional=False, name=None):

        assert len(layers) == 2, self.__class__.__name__ + " layers argument must be length 2 list"

        super(Seq2Seq, self).__init__(name=name)

        def get_container(l, cls):
            if isinstance(l, cls):
                return l
            elif isinstance(l, list):
                return cls(l)
            elif isinstance(l, Layer):
                return cls([l])
            else:
                ValueError("Incompatible element for " + self.__class__.__name__ + " container")

        self.encoder = get_container(layers[0], Encoder)
        self.decoder = get_container(layers[1], Decoder)
        self.layers = self.encoder.layers + self.decoder.layers

        self.conditional = conditional
        self.hasLUT = isinstance(self.encoder.layers[0], LookupTable)

        if decoder_connections:
            self.decoder_connections = decoder_connections
        else:
            # if decoder_connections not given, assume one to one connections between
            # an equal number of encoder and decoder recurrent layers
            assert len(self.encoder._recurrent) == len(self.decoder._recurrent)
            self.decoder_connections = np.arange(len(self.encoder._recurrent)).tolist()

        self.encoder.set_connections(self.decoder_connections)
        self.decoder.set_connections(self.decoder_connections)

    @classmethod
    def gen_class(cls, pdict):
        layers = [[], []]
        for i, layer in enumerate(pdict['layers']):
            typ = layer['type']
            ccls = load_class(typ)

            if i < pdict['num_encoder_layers']:
                layers[0].append(ccls.gen_class(layer['config']))
            else:
                layers[1].append(ccls.gen_class(layer['config']))

        # layers is special in that there may be parameters
        # serialized which will be used elsewhere
        lsave = pdict.pop('layers')
        pdict.pop('num_encoder_layers', None)
        new_cls = cls(layers=layers, **pdict)
        pdict['layers'] = lsave
        return new_cls

    def get_description(self, get_weights=False, keep_states=False):
        """
        Get layer parameters. All parameters are needed for optimization, but
        only weights are serialized.

        Arguments:
            get_weights (bool, optional): Control whether all parameters are returned or
                                          just weights for serialization.
            keep_states (bool, optional): Control whether all parameters are returned
                                          or just weights for serialization.
        """
        desc = super(Seq2Seq, self).get_description(get_weights=get_weights,
                                                    keep_states=keep_states)

        desc['config']['num_encoder_layers'] = len(self.encoder.layers)
        desc['config']['decoder_connections'] = self.decoder_connections
        desc['config']['conditional'] = self.conditional
        self._desc = desc
        return desc

    def configure(self, in_obj):
        # assumes Seq2Seq will always get dataset as in_obj
        if not self.conditional:
            self.encoder.configure(in_obj)
            self.decoder.configure(in_obj)
        else:
            assert in_obj.conditional is True, \
                   "In conditional AE case, the input should have 2 elements"
            self.encoder.configure(in_obj.shape[0])
            self.decoder.configure(in_obj.shape[1])
        self.parallelism = self.decoder.parallelism
        self.out_shape = self.decoder.out_shape
        self.in_shape = self.decoder.layers[-1].in_shape if self.hasLUT else self.decoder.in_shape
        # save full sequence length for switching between inference and non-inference modes
        self.decoder.full_steps = self.in_shape[1]
        return self

    def allocate(self, shared_outputs=None):
        self.decoder.allocate(shared_outputs)
        if any([l.owns_output for l in self.decoder.layers]):
            self.encoder.allocate()
        else:
            self.encoder.allocate(shared_outputs)
        # buffer for collecting time loop outputs
        self.xbuf = self.be.iobuf(self.out_shape)

    def allocate_deltas(self, global_deltas=None):
        self.encoder.allocate_deltas(global_deltas)
        self.decoder.allocate_deltas(global_deltas)

    def fprop(self, inputs, inference=False, beta=0.0):
        if not (inference and self.conditional):

            # make sure we are in the correct decoder mode
            self.decoder.switch_mode(inference, self.conditional)

            # load data
            if self.conditional:
                (x, z) = inputs
            else:
                x = inputs
                z = x.backend.zeros(x.shape)

            # fprop through Encoder layers
            x = self.encoder.fprop(x, inference=inference, beta=0.0)

            # get encoder hidden state
            init_state_list = self.encoder.get_final_states(self.decoder_connections)

            # fprop through Decoder layers
            x = self.decoder.fprop(z, inference=inference, init_state_list=init_state_list)
        else:

            # Loopy inference.
            self.decoder.switch_mode(inference, self.conditional)

            # prep data
            x = inputs
            new_steps = 1
            if self.hasLUT:
                z_shape = new_steps
            else:
                z_shape = (self.out_shape[0], new_steps)
            z = x.backend.iobuf(z_shape)

            # encoder
            x = self.encoder.fprop(x, inference=inference, beta=0.0)

            # get encoder hidden state
            init_state_list = self.encoder.get_final_states(self.decoder_connections)

            # decoder
            steps = self.in_shape[1]
            if self.hasLUT:
                z_argmax = x.backend.zeros((1, z.shape[0]*z.shape[1]))

            for t in range(steps):
                z = self.decoder.fprop(z, inference=inference, init_state_list=init_state_list)

                # transfer hidden state from DECODER to next step
                init_state_list = [recurrent.final_state()
                                   for recurrent in self.decoder._recurrent]

                # and write to output buffer
                self.xbuf[:, t*self.be.bsz:(t+1)*self.be.bsz] = z

                # handle input to LUT
                if self.hasLUT:
                    z_argmax[:] = self.be.argmax(z, axis=0)
                    z = z_argmax

            x = self.xbuf

        if inference:
            self.revert_tensors()

        return x

    def bprop(self, error, inference=False, alpha=1.0, beta=0.0):
        hidden_error_list = self.decoder.bprop(error)
        self.encoder.bprop(hidden_error_list)

        return self.encoder.layers[0].deltas


class Multicost(NervanaObject):
    """
    Class used to compute cost from a Tree container with multiple outputs.
    The number of costs must match the number of outputs.  Costs will be applied to the outputs
    in the same order that they occur in the Tree.

    The targets used for the cost can either be provided from the dataset as a list or tuple,
    one for each cost, or, if only a single target is provided, the same target is used for all
    costs.  This is useful for providing multiple cost branches computing the same error at
    different stages of the network as in GoogLeNet.
    """

    def __init__(self, costs, weights=None, name=None):
        super(Multicost, self).__init__(name)
        self.costs = costs
        self.weights = [1.0 for c in costs] if weights is None else weights
        self.errors = None
        self.inputs = None
        self.costfunc = costs[0].costfunc  # For displaying during callbacks

    def initialize(self, in_obj):
        """
        Determine dimensions of cost and error buffers and allocate space from the input layer

        Arguments:
            in_obj (Layer): input layer from which to calculate costs
        """
        assert hasattr(in_obj, 'layers'), "MultiCost must be passed a layer container"
        terminals = in_obj.get_terminal()
        for c, ll in zip(self.costs, terminals):
            c.initialize(ll)

    @property
    def cost(self):
        """ Get cost. """
        return self.costs[0].cost

    @property
    def outputs(self):
        """ Get outputs. """
        return self.costs[0].outputs

    def get_description(self, **kwargs):
        """
        Get layer parameters.

        Arguments:
            **kwargs: ignored
        """
        desc = super(Multicost, self).get_description()
        costs = desc['config'].pop('costs')
        desc['config']['costs'] = []
        for cost in costs:
            desc['config']['costs'].append(cost.get_description())
        self._desc = desc
        return desc

    def get_cost(self, inputs, targets):
        """
        Compute the cost function over a list of inputs and targets.

        Arguments:
            inputs (list(Tensor)): list of Tensors containing input values to be compared to
                                   targets
            targets (Tensor, list(Tensor)): either a list of Tensors containing target values, or
                                            a single target Tensor that will be mapped to each
                                            input

        Returns:
            Tensor containing cost
        """
        if not isinstance(inputs, list):
            return self.costs[0].get_cost(inputs, targets)
        else:
            ltargets = targets if type(targets) in (tuple, list) else [targets for c in self.costs]
            costvals = [c.get_cost(i, t) for c, i, t in zip(self.costs, inputs, ltargets)]
            sum_optree = reduce(add, [w * c for w, c in zip(self.weights, costvals)])
            costvals[0][:] = sum_optree
            return costvals[0]

    def get_errors(self, inputs, targets):
        """
        Get a list of errors for backpropagating to a Tree container that has multiple output
        nodes.

        Arguments:
            inputs (list(Tensor)): list of Tensors containing input values to be compared to
                                   targets
            targets (Tensor, list(Tensor)): either a list of Tensors containing target values, or
                                            a single target Tensor that will be mapped to each
                                            input
        Returns:
            list of Tensors containing errors for each input
        """

        l_targets = targets if type(targets) in (tuple, list) else [targets for c in self.costs]
        if self.errors is None:
            self.errors = [c.deltas for c in self.costs]

        for c, i, t in zip(self.costs, inputs, l_targets):
            c.get_errors(i, t)

        return self.errors
