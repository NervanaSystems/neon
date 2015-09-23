from neon.layers.layer import Layer
import itertools
import numpy as np


class Merge(Layer):
    """
    Merge layer which merges the outputs of multiple input models where each
    input model is a list of Layers or a Layer

    Attributes:
        layer_container (list): List of objects which can be either a list of
                                layers or a layer.  Each element in
                                layer_container is a separate model which runs
                                on its own input.
        deltas (list): Used by MergeConcat. List of deltas (Tensors) to pass
                       to each input model during bprop.
    """
    def __init__(self, layer_container, name='merge'):
        assert len(layer_container) > 1
        super(Merge, self).__init__(name)
        self.deltas = None
        self.layers_to_optimize = []

        list_container = [obj if isinstance(obj, list) else [obj] for obj in layer_container]

        def flatten(layer_list):
            ll = [x if isinstance(x, list) else [x] for x in layer_list]
            return list(itertools.chain.from_iterable(ll))

        # Flatten out each list in layer_container
        self.layer_container = [flatten(layer_list) for layer_list in list_container]

        for lobj in self.layer_container:
            param_layers = [l for l in lobj if (l.has_params and isinstance(l, Layer))]
            self.layers_to_optimize.extend(param_layers)

    def configure(self, in_obj):
        """
        Must receive a list of shapes for configuration (one for each pathway)
        the shapes correspond to the layer_container attribute

        Arguments:
            in_obj (list)
        """
        if not isinstance(in_obj, list):
            assert hasattr(in_obj, 'shape') and isinstance(in_obj.shape, list)
            in_obj = in_obj.shape
        assert isinstance(in_obj, list), "Must provide a list of shapes to Merge configure"
        self.in_shape = []
        for inp, lobj in zip(in_obj, self.layer_container):
            for l in lobj:
                inp = l.configure(inp)
            self.in_shape.append(inp.out_shape)
        return self
        # self.out_shape will be determined by merge type

    def _do_fprop(self, obj, inputs, inference=False):
        """
        Helper fprop function which calls fprop on a given input model.

        Arguments:
            obj (list): input model to call fprop on.  Must be a list of Layers
            inputs (Tensor): Input tensor data to fprop on.
            inference (bool): Flag if doing inference or not

        Returns:
            Tensor: output data
        """
        assert isinstance(obj, list), "Merge Layers only operate on lists of Layers."
        tmp_output = inputs
        for layer in obj:
            tmp_output = layer.fprop(tmp_output, inference)
        return tmp_output

    def _do_bprop(self, obj, error):
        """
        Helper bprop function which calls bprop on a given input model

        Arguments:
            obj (list): input model to call bprop on. Must be a list of Layers
            error (tensor): Error tensor to bprop on

        """

        assert isinstance(obj, list), "Merge Layers only operate on lists of Layers."
        tmp_deltas = error
        for layer in reversed(obj[1:]):
            tmp_deltas = layer.bprop(tmp_deltas)
        obj[0].bprop(tmp_deltas, do_acts=False)

    def fprop(self, inputs, inference=False):
        """
        Called by fprop of sublcasses of Merge. Checks that we're merging more than
        one model and that the length of inputs and layer_container are equal.
        Calls fprop on each input model on inputs.  Assumes inputs list is
        aligned with layer_container list.

        Arguments:
            inputs (list): list of input Tensors
            inference: (bool): flag if doing inference or not

        Returns:
            Tensor: output data
        """
        assert len(inputs) > 1
        assert len(inputs) == len(self.layer_container), "Input / Layer mismatch."

        for lc, x in zip(self.layer_container, inputs):
            self._do_fprop(lc, x, inference)

        return self.outputs

    def bprop(self, error, do_acts=False):
        """
        Abstract bprop method.

        Arguments:
            error (Tensor): Single error tensor to bprop on.
        """
        for lc, delta in zip(self.layer_container, self.deltas):
            self._do_bprop(lc, delta)

    def get_description(self):
        desc = super(Merge, self).get_description()
        layer_container_desc = [[l.get_description() for l in ll] for ll in self.layer_container]
        desc['layer_container'] = layer_container_desc
        return desc


class MergeSum(Merge):
    """
    Merge layer that sums the outputs of each input model
    """

    def configure(self, in_obj):
        """
        By default, retain local shape if one of the pathways is a local layer
        """
        super(MergeSum, self).configure(in_obj)
        flatdims = [np.prod(xs) if isinstance(xs, tuple) else xs for xs in self.in_shape]
        assert flatdims[1:] == flatdims[:-1], "MergeSum elements must have same number of outputs"
        self.flatdim = flatdims[0]
        self.out_shape = self.in_shape[0]
        return self.out_shape

    def allocate(self, shared_outputs=None, shared_deltas=None):
        self.outputs = self.be.iobuf(self.flatdim)
        self.optree_sum = 0
        for lobj in self.layer_container:
            for l in lobj:
                l.allocate()
            # cache the summation via optree to avoid multiple kernel calls
            self.optree_sum = self.optree_sum + lobj[-1].outputs

    def fprop(self, inputs, inference=False):
        super(MergeSum, self).fprop(inputs, inference)
        self.outputs[:] = self.optree_sum
        return self.outputs

    def bprop(self, error, do_acts=False):
        if self.deltas is None:  # This is just to set up for using the parent bprop
            self.deltas = [error for i in self.layer_container]
        super(MergeSum, self).bprop(error, do_acts)


class MergeConcat(Merge):
    """
    Merge layer that concatenates (on the first dimension) the outputs of each input model.
    For example if input1 is (f1,n) and input2 is (f2, n), then the result will be (f1+f2, n)
    where f1 and f2 are the feature sizes of each input and n is the batch size.
    """

    def __init__(self, layer_container, name="MergeConcatLayer"):
        super(MergeConcat, self).__init__(layer_container, name=name)
        self.concat_deltas_shape = None

    def configure(self, in_obj):
        super(MergeConcat, self).configure(in_obj)
        flatdims = [xs if isinstance(xs, int) else np.prod(xs) for xs in self.in_shape]
        self.out_shape = sum(flatdims)
        end_idx = [idx for idx in np.cumsum(flatdims)]
        start_idx = [0] + end_idx[:-1]
        self.ranges = [slice(s, e) for s, e in zip(start_idx, end_idx)]
        return self.out_shape

    def allocate(self, shared_outputs=None, shared_deltas=None):
        self.outputs = self.be.iobuf(self.out_shape)
        for lobj, view in zip(self.layer_container, self.ranges):
            for l in lobj[:-1]:
                l.allocate()
            lobj[-1].allocate(shared_outputs=self.outputs[view])

    def bprop(self, error, do_acts=False):
        if self.deltas is None:
            self.deltas = [error[view] for view in self.ranges]
        super(MergeConcat, self).bprop(error, do_acts)


class MergeConcatSequence(Merge):
    """
    Merge layer that concatenates (on the second dimension) the outputs of each input model.
    in_shape for sequence input are 2d (feature_size, nsteps)
    so if we have input 1 that is (f, n) and input 2 that is (f, m), then the output will be
    (f, n + m)
    """

    def configure(self, in_obj):
        super(MergeConcatSequence, self).configure(in_obj)
        for xs in self.in_shape:
            assert(isinstance(xs, tuple) and len(xs) == 2), "Not valid sequence input"
        stepdims = [xs[1] for xs in self.in_shape]
        self.out_shape = (self.in_shape[0][0], sum(stepdims))
        end_idx = [idx * self.be.bsz for idx in np.cumsum(stepdims)]
        start_idx = [0] + end_idx[:-1]
        self.ranges = [slice(s, e) for s, e in zip(start_idx, end_idx)]
        return self.out_shape

    def allocate(self, shared_outputs=None, shared_deltas=None):
        for lobj in self.layer_container:
            for l in lobj:
                l.allocate()
        self.outputs = self.be.iobuf(self.out_shape)
        self.deltas = None
        self.output_views = [self.outputs[:, view] for view in self.ranges]
        self.x_views = [lobj[-1].outputs for lobj in self.layer_container]

    def fprop(self, inputs, inference=False):
        super(MergeConcatSequence, self).fprop(inputs, inference)
        for output, x in zip(self.output_views, self.x_views):
            output[:] = x
        return self.outputs

    def bprop(self, error, do_acts=False):
        if self.deltas is None:
            self.deltas = [error[:, view] for view in self.ranges]
        super(MergeConcatSequence, self).bprop(error, do_acts)
