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

from __future__ import division
from builtins import str, zip, range
import numpy as np
import itertools as itt
from operator import add

from neon import NervanaObject
from neon.layers.layer import Layer, BranchNode, Dropout, DataTransform, LookupTable, Affine
from neon.layers.recurrent import Recurrent, get_steps
from neon.transforms import Softmax
from neon.util.persist import load_class
from functools import reduce
from funcsigs import signature


# modified from https://docs.python.org/3/library/itertools.html
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..., (sN, None)"
    a, b = itt.tee(iterable + [None])
    next(b, None)
    return zip(a, b)


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
    def __init__(self, name=None):
        super(LayerContainer, self).__init__(name)
        self.is_mklop = True

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

    def fusion_pass(self, layers):
        """
        Groups patterns together in list. If pattern is [a, b], will transform
        [a, b, c, d, a, b, e] -> [[a, b], c, d, [a, b], e]. Support for multiple
        patterns.
        """
        patterns = [lambda x, y: x['type'] == 'neon.layers.layer.Convolution' and
                    y['type'] == 'neon.layers.layer.Bias']

        result = []
        skip_next = False
        for (l1, l2) in pairwise(layers):
            if any([pattern(l1, l2) for pattern in patterns]):
                result.append([l1, l2])
                skip_next = True
            elif skip_next:
                skip_next = False
            else:
                result.append(l1)

        return result

    def load_weights(self, pdict, load_states=True):
        """
        Load weights.

        Arguments:
            pdict:
            load_states:  (Default value = True)

        Returns:

        """
        pdict['config']['layers'] = self.fusion_pass(pdict['config']['layers'])

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

    def set_acc_on(self, acc_on):
        """
        Set the acc_on flag according to bool argument for each layer.
        If a layer in the container does not support accumulate_updates
        it will be skipped.

        Arguments:
           acc_on (bool): Value to set the acc_on flag of supported layers to.
        """
        if (not hasattr(self, "accumulate_updates")):
            raise BufferError("accumulate_updates not set")
        for l in self.layers:
            if hasattr(l, "accumulate_updates"):
                l.set_acc_on(acc_on)


class Sequential(LayerContainer):
    """
    Layer container that encapsulates a simple linear pathway of layers.

    Arguments:
        layers (list): List of objects which can be either a list of layers
                       (including layer containers).
    """
    def __init__(self, layers, name=None):
        super(Sequential, self).__init__(name)

        assert layers, "Provide layers"
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

    def allocate(self, shared_outputs=None, accumulate_updates=False):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
        """
        # get the layers that own their outputs
        self.accumulate_updates = accumulate_updates
        alloc_layers = [l for l in self.layers if l.owns_output]
        if 'accumulate_updates' in signature(alloc_layers[-1].allocate).parameters:
            alloc_layers[-1].allocate(shared_outputs, accumulate_updates=accumulate_updates)
        else:
            alloc_layers[-1].allocate(shared_outputs)
        for l in self.layers:
            if 'accumulate_updates' in signature(l.allocate).parameters:
                l.allocate(accumulate_updates=accumulate_updates)
            else:
                l.allocate()

    def allocate_deltas(self, global_deltas=None):
        if global_deltas is None:
            self.global_deltas = DeltasTree()

            st_ind = 0 if getattr(self.layers[0], 'nest_deltas', False) else 1
            for layer in self.layers[st_ind:]:
                layer.allocate_deltas(self.global_deltas)

            self.global_deltas.allocate_buffers()
        else:
            self.global_deltas = global_deltas

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

            # try to convert to mkl
            l.be.convert_data(x, l.get_is_mklop())

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

            # try to convert to mkl
            l.be.convert_data(error, l.get_is_mklop())
            if l.deltas is not None:
                l.be.clean_data(l.deltas, l.get_is_mklop())

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


class GenerativeAdversarial(Sequential):
    """
    Container for Generative Adversarial Net (GAN). It contains the Generator
    and Discriminator stacks as sequential containers.

    Arguments:
        layers (list): A list containing two Sequential containers
    """
    def __init__(self, generator, discriminator, name=None):
        super(Sequential, self).__init__(name)

        self.generator = generator
        self.discriminator = discriminator
        self.layers = self.generator.layers + self.discriminator.layers

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
        ss += '  ' * level + 'Generator:\n'
        ss += padstr.join([l.nested_str(level + 1) for l in self.generator.layers])
        ss += '\n' + '  ' * level + 'Discriminator:\n'
        ss += padstr.join([l.nested_str(level + 1) for l in self.discriminator.layers])
        return ss


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

    def bprop(self, error, alpha=1.0, beta=0.0):
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

            # mkl need allocate new deltas
            l.layers[0].deltas = self.be.allocate_new_deltas(
                l.layers[0].deltas, l.layers[0].in_shape, l.layers[0].parallelism)

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
    def __init__(self, layers, name=None):
        super(MergeSum, self).__init__(layers, name)
        self.ngLayer = self.be.mergesum_layer(len(layers))

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
            self.be.allocate_new_outputs(l, self.outputs)

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
        self.be.fprop_mergesum(self.ngLayer, inputs, inference,
                               self.layers, self.outputs, self.out_shape)
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
        self.be.bprop_mergesum(self.ngLayer, alpha, beta,
                               self.layers, error, self.deltas)
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
        self.ngLayer = self.be.mergebroadcast_layer(len(layers))

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
        self.be.fprop_mergebroadcast(
            self.ngLayer, inputs, inference, self.outputs,
            self.layers, self.out_shape)
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
        self.be.bprop_mergebroadcast(
            self.ngLayer, self.layers, self.error_views, error,
            self.deltas, self.out_shape, alpha, beta, self.alphas, self.betas)
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
    def __init__(self, layers, name=None, start_index=None):
        """
        Arguments:
            layers: the layers to use for this Decoder
            start_index: the index of the symbol to use as the start symbol
                         when generating a new sequence.  If None, defaults to
                         all 0 (not the same as index 0)

        """
        super(Decoder, self).__init__(layers, name)
        # list of recurrent layers only:
        self._recurrent = [l for l in self.layers if isinstance(l, Recurrent)]
        self.connections = None
        self.full_steps = None
        self.start_index = start_index

    def fprop(self, x, inference=False, init_state_list=None):

        if init_state_list is None:
            init_state_list = [None for _ in range(len(self._recurrent))]
        else:
            if len(self._recurrent) > len(init_state_list):
                raise ValueError((
                    'found {n} Recurrent layers, but init_state_list is only length {l}'
                ).format(
                    n=len(self._recurrent),
                    l=len(init_state_list),
                ))

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

    def switch_mode(self, inference):
        """
        Dynamically grow or shrink the number of time steps to perform
        single time step fprop during inference.
        """
        # set up parameters
        hasLUT = isinstance(self.layers[0], LookupTable)

        # sequence length is different dimension depending on whether there is LUT
        cur_steps = self.in_shape[0] if hasLUT else self.in_shape[1]
        if not inference:
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
                self.layers[0].inputs = None  # ensure "allocate" will reallocate this buffer
                self.layers[0].outputs_t = None
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
    """
    def __init__(self, layers, decoder_connections=None, name=None):

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
        self.prev_layer = None

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
        self._desc = desc
        return desc

    def configure(self, in_obj):
        if isinstance(in_obj, tuple):
            assert len(in_obj) == 4, "Seq2Seq requires in_obj to provide shape and decoder_shape"
            # assumes deserialized input with both shapes concatenated into in_obj
            # necessary to support serialization into train_input_shape for cloud inference
            self.encoder.configure(in_obj[0:2])
            self.decoder.configure(in_obj[2:4])
        else:
            # assumes Seq2Seq will always get dataset as in_obj
            self.encoder.configure(in_obj.shape)
            self.decoder.configure(in_obj.decoder_shape)

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
        """
        Forward propagation for sequence to sequence container. Calls
        fprop for the Encoder container followed by fprop for the Decoder
        container. If inference is True, the Decoder will be called with
        individual time steps in a for loop.
        """
        # make sure we are in the correct decoder mode
        self.decoder.switch_mode(inference)

        if not inference:
            # load data
            (x, z) = inputs

            # fprop through Encoder layers
            x = self.encoder.fprop(x, inference=inference, beta=0.0)

            # get encoder hidden state
            init_state_list = self.encoder.get_final_states(self.decoder_connections)

            # fprop through Decoder layers
            x = self.decoder.fprop(z, inference=inference, init_state_list=init_state_list)
        else:  # Loopy inference

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
        """
        Backpropagation for sequence to sequence container. Calls Decoder container
        bprop followed by Encoder container bprop.
        """

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
        self.deltas = None
        self.inputs = None
        self.costfunc = costs[0].costfunc  # For displaying during callbacks

    def initialize(self, in_obj):
        """
        Determine dimensions of cost and error buffers and allocate space from the input layer

        Arguments:
            in_obj (Layer): input layer from which to calculate costs
        """
        if isinstance(in_obj, LayerContainer):
            terminals = in_obj.get_terminal()
        elif isinstance(in_obj, list):
            terminals = in_obj
        else:
            raise RuntimeError("Multicost must be passed a container or list")

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
            if type(targets) not in (tuple, list):
                targets = [targets] * len(self.costs)

            costs = []
            for w, c, i, t in zip(self.weights, self.costs, inputs, targets):
                # TODO: use sentinal class instead of None

                # it is important that we don't even call get_cost on costs
                # which aren't applicable because there are hooks and state
                # that get set that we don't want to include.
                if t is not None:
                    costs.append(w * c.get_cost(i, t))

            return reduce(add, costs)

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
        if type(targets) not in (tuple, list):
            targets = [targets] * len(self.costs)

        for cost, i, t, we in zip(self.costs, inputs, targets, self.weights):
            if t is None:
                continue

            cost.get_errors(i, t)
            if isinstance(cost.deltas, list):
                for delta in cost.deltas:
                    delta[:] *= we
            else:
                cost.deltas[:] *= we

        if self.deltas is None:
            self.deltas = [c.deltas for c in self.costs]

        return self.deltas


class SkipThought(Sequential):

    """
    A skip-thought container that encapsulates the network architectue:
                                       ,--> Previous Sentence
        Source sentence --> Embedding <
                                       `--> Next Sentence
    Arguments:
        vocab_size: vocabulary size
        embed_dim: word embedding dimension
        init_embed: word embedding initialization
        nhidden: number of hidden units
        rec_layer: recurrent layer type to use for encoder and decoder (GRU or LSTM)
        init_rec: initializer to use for recurrent connections
        activ_rec: activation function to use for recurrent connections
        activ_rec_gate: activation function to use for gated connections in recurrent layer
        init_ff: initializer to use for final decoder feed forward layers
        init_const: constant initializer to use for biases
    """

    def __init__(self, vocab_size, embed_dim, init_embed, nhidden,
                 rec_layer, init_rec, activ_rec, activ_rec_gate,
                 init_ff, init_const, layers=None, name=None):

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.init_embed = init_embed
        self.nhidden = nhidden
        self.owns_output = True
        self.owns_delta = True

        self.rec_layer = rec_layer
        self.init_rec = init_rec
        self.activ_rec = activ_rec
        self.activ_rec_gate = activ_rec_gate

        self.init_ff = init_ff
        self.init_const = init_const

        if layers is None:
            self.embed_s = LookupTable(vocab_size=vocab_size, embedding_dim=embed_dim,
                                       init=init_embed, pad_idx=0)
            self.embed_p = LookupTable(vocab_size=vocab_size, embedding_dim=embed_dim,
                                       init=init_embed, pad_idx=0)
            self.embed_n = LookupTable(vocab_size=vocab_size, embedding_dim=embed_dim,
                                       init=init_embed, pad_idx=0)

            self.encoder = rec_layer(nhidden, init=init_ff, init_inner=init_rec,
                                     activation=activ_rec, gate_activation=activ_rec_gate,
                                     reset_cells=True)
            self.decoder_p = rec_layer(nhidden, init=init_ff, init_inner=init_rec,
                                       activation=activ_rec, gate_activation=activ_rec_gate,
                                       reset_cells=True)
            self.decoder_n = rec_layer(nhidden, init=init_ff, init_inner=init_rec,
                                       activation=activ_rec, gate_activation=activ_rec_gate,
                                       reset_cells=True)

            self.affine_p = Affine(
                vocab_size, init=init_ff, bias=init_const, activation=Softmax())
            self.affine_n = Affine(
                vocab_size, init=init_ff, bias=init_const, activation=Softmax())

            self.layers = [self.embed_s, self.embed_p, self.embed_n, self.encoder,
                           self.decoder_p, self.decoder_n] + self.affine_p + self.affine_n

            # Create a layer dict to re-load the model for evaluation
            self.layer_dict = dict()
            self.layer_dict['lookupTable'] = self.embed_s
            self.layer_dict['encoder'] = self.encoder
            self.layer_dict['decoder_previous'] = self.decoder_p
            self.layer_dict['decoder_next'] = self.decoder_n
            self.layer_dict['affine'] = self.affine_p
        else:
            assert len(layers) == 12
            self.layers = layers

            # Create a layer dict to re-load the model for evaluation
            self.layer_dict = dict()
            self.layer_dict['lookupTable'] = self.layers[0]
            self.layer_dict['encoder'] = self.layers[3]
            self.layer_dict['decoder_previous'] = self.layers[4]
            self.layer_dict['decoder_next'] = self.layers[5]
            self.layer_dict['affine'] = self.layers[6:9]

        super(SkipThought, self).__init__(layers=self.layers, name=name)

    def allocate(self, shared_outputs=None):
        """
        Allocate backend memory for dW's. Sync initial affine layer weights.
        """
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
        word embedding layer first.
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

    def fprop(self, inputs, inference=False):
        """
        Encode source sentence and use embedding to initialize state of two decoders to predict the
        next and previous sentences.

        Arguments:
            inputs (list): Length 3 list of [source_sentence, previous_sentence, next_sentence]
            inference (bool): Not implemented.
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
        Backpropagate the error from both output branches (forward and backward, 2 elements).
        Sync the dW's of the word-embedding layers, and the feed-forward output layers.

        Arguments:
            error (list): error from previous sentence reconstruction,
                          error from next sentence reconstruction
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
