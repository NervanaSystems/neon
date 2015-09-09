from neon.layers.layer import Layer


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
        self.layer_container = []
        self.deltas = None
        self.layers_to_optimize = []

        # Flatten out each list in layer_container
        for obj in layer_container:
            if isinstance(obj, list):
                path_layers = []
                for l in obj:
                    if isinstance(l, list):
                        path_layers.extend(l)
                    else:
                        path_layers.append(l)
                self.layer_container.append(path_layers)
            else:
                self.layer_container.append(obj)

        for obj in self.layer_container:
            if isinstance(obj, list):
                for l in obj:
                    if l.has_params:
                        self.layers_to_optimize.append(l)
            elif isinstance(obj, Layer):
                if obj.has_params:
                    self.layers_to_optimize.append(obj)

    def _do_fprop(self, obj, inputs, inference=False):
        """
        Helper fprop function which calls fprop on a given input model.

        Arguments:
            obj (list or Layer): input model to call fprop on.  Can be a list
                                 of Layers or a Layer
            inputs (Tensor): Input tensor data to fprop on.
            inference (bool): Flag if doing inference or not

        Returns:
            Tensor: output data
        """

        if isinstance(obj, Layer):
            return obj.fprop(inputs, inference)

        elif isinstance(obj, list):
            tmp_output = inputs
            for layer in obj:
                tmp_output = layer.fprop(tmp_output, inference)
            return tmp_output
        else:
            raise Exception("Merge Layers only operate on Layers and lists of Layers.")

    def _do_bprop(self, obj, error):
        """
        Helper bprop function which calls bprop on a given input model

        Arguments:
            obj (list or Layer): input model to call fprop on. Can be a list
                                 of Layers or a Layer
            error (tensor): Error tensor to bprop on

        """

        if isinstance(obj, Layer):
            return obj.bprop(error, do_acts=False)

        elif isinstance(obj, list):
            tmp_deltas = error
            for layer in reversed(obj[1:]):
                tmp_deltas = layer.bprop(tmp_deltas)
            obj[0].bprop(tmp_deltas, do_acts=False)
        else:
            raise Exception("Merge Layers only operate on Layers and lists of Layers")

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

    def bprop(self, error, do_acts=False):
        """
        Abstract bprop method.

        Arguments:
            error (Tensor): Single error tensor to bprop on.
        """

        raise NotImplementedError

    def get_description(self):
        desc = super(Merge, self).get_description()
        layer_container_desc = []
        for obj in self.layer_container:
            if isinstance(obj, Layer):
                layer_container_desc.append(obj.get_description())
            elif isinstance(obj, list):
                layer_container_desc.append([l.get_description() for l in obj])

        desc['layer_container'] = layer_container_desc
        return desc


class MergeSum(Merge):
    """
    Merge layer that sums the outputs of each input model
    """

    def fprop(self, inputs, inference=False):
        # call fprop on the first element of the layer container to
        # infer the size of our new sum buffer from the outputs.
        # then create this buffer before calling fprop on the rest of the
        # layer_container.

        first_output = self._do_fprop(self.layer_container[0], inputs[0], inference)
        if self.outputs is None:
            self.outputs = self.be.empty_like(first_output)
        self.outputs[:] = first_output

        for i in range(1, len(self.layer_container)):
            self.outputs[:] = (self.outputs +
                               self._do_fprop(self.layer_container[i],
                                              inputs[i], inference))

        return self.outputs

    def bprop(self, error, do_acts=False):
        for obj in self.layer_container:
            self._do_bprop(obj, error)


class MergeConcat(Merge):
    """
    Merge layer that concatenates (on the first dimension) the outputs of each input model.
    For example if input1 is (f1,n) and input2 is (f2, n), then the result will be (f1+f2, n)
    where f1 and f2 are the feature sizes of each input and n is the batch size.
    """

    def __init__(self, layer_container, name="MergeConcatLayer"):
        super(MergeConcat, self).__init__(layer_container, name=name)
        self.concat_deltas_shape = None

    def fprop(self, inputs, inference=False):
        outputs_list = [self._do_fprop(lc, x, inference)
                        for lc, x in zip(self.layer_container, inputs)]
        dims = [x.shape[1:] for x in outputs_list]
        assert len(set(dims)) == 1, ("Elements of Merge Layer must have equal "
                                     "not-first dimensions in order to "
                                     "concatenate.")
        # Always concat on first dimension
        # create the output buffer and a flattened view
        # that makes the concatenation easier
        if self.outputs is None:
            # compute the correct output shape
            nout_shape = tuple([sum([x.shape[0] for x in outputs_list])] +
                               [dim for dim in outputs_list[0].shape[1:]])
            self.concat_deltas_attr = [(x.size, x.shape) for x in outputs_list]
            self.outputs = self.be.zeros(nout_shape)
            self.flat_outputs_view = self.outputs.reshape(self.outputs.size)
        start = 0
        for x in outputs_list:
            self.flat_outputs_view[start:start+x.size][:] = x.reshape(x.size)
            start += x.size
        return self.outputs

    def bprop(self, error, do_acts=False):
        if self.deltas is None:
            # initialize as flattened using size
            self.deltas = [self.be.zeros(x[0]) for x in self.concat_deltas_attr]
        error = error.reshape(error.size)
        start = 0
        for delta in self.deltas:
            delta[:] = error[start:start+delta.size]
            start += delta.size
        for (layer, delta, x) in zip(self.layer_container, self.deltas, self.concat_deltas_attr):
            self._do_bprop(layer, delta.reshape(x[1]))


class MergeConcatSequence(Merge):
    """
    Merge layer that concatenates (on the second dimension) the outputs of each input model.
    For example if input1 is (f,n) and input2 is (f,m) then the result will be (f,n+m) where
    f is the feature size, and n and m are steps * batch_size.
    """

    def fprop(self, inputs, inference=False):
        outputs_list = [self._do_fprop(lc, x, inference)
                        for lc, x in zip(self.layer_container, inputs)]
        assert len(outputs_list[0].shape) == 2, "Can only concat 2d inputs"

        if self.outputs is None:
            self.inputs_shape = [x.shape for x in outputs_list]
            nout_shape = [self.inputs_shape[0][0]] + [sum(x[1] for x in self.inputs_shape)]
            self.x_view = [x.T for x in outputs_list]

            self.outputs = self.be.zeros(nout_shape)
            self.outputs_view = []
            start = 0
            for x in outputs_list:
                self.outputs_view.append(self.outputs[:, start:start+x.shape[1]].T)
                start += x.shape[1]

        for output, x in zip(self.outputs_view, self.x_view):
            output[:] = x

        return self.outputs

    def bprop(self, error, do_acts=False):
        if self.deltas is None:
            self.deltas = [self.be.zeros(x) for x in self.inputs_shape]
            self.deltas_view = [x.T for x in self.deltas]

            self.errors_view = []
            start = 0
            for x in self.deltas:
                self.errors_view.append(error[:, start:start+x.shape[1]].T)
                start += x.shape[1]

        for delta, e in zip(self.deltas_view, self.errors_view):
            delta[:] = e

        for (layer, delta) in zip(self.layer_container, self.deltas):
            self._do_bprop(layer, delta)
