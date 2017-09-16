from __future__ import division
import numpy as np
from neon.data.dataloader_transformers import DataLoaderTransformer
from collections import OrderedDict


class DataLoaderAdapter(DataLoaderTransformer):
    """
    DataLoaderAdapter converts Aeon data buffers to tensors.

    Arguments:
        dataloader (DataLoader): Aeon dataloading module.
    """
    def __init__(self, dataloader):
        super(DataLoaderAdapter, self).__init__(dataloader, None)

        self.shape = self.shapes()[0]
        self.nbatches, modal = divmod(self.dataloader.ndata, self.be.bsz)
        if modal > 0:
            self.nbatches += 1
        self.outputs = OrderedDict()  # Create output buffers

        def max_dur(val, freq):
            uval = float(val.split(" ")[0])
            ucat = val.split(" ")[1]
            if(ucat == "samples"):
                return float(uval)
            elif(ucat == "seconds"):
                return float(uval * freq)
            elif(ucat == "milliseconds"):
                return float(uval/1000 * freq)
            else:
                raise ValueError("Unknown time unit " + ucat)

        # Convert max duration value for audio from Aeon format
        for conf in self.dataloader.config['etl']:
            if(conf['type'] == "audio"):
                self.max_duration = max_dur(conf['max_duration'], conf['sample_freq_hz'])

    def transform(self, t):
        """
        Converts Aeon data to tuple of tensors.

        Arguments:
            t (tuple): Tuple of numpy arrays.
                For example: {tuple}{'image':ndarray(...), 'label':ndarray(...)}
                where 'image' shape is (N,C,H,W) and 'label' shape is (N,1)
        """
        for key, value in t:
            assert value.shape[0] == self.be.bsz

            reshape_rows = self.be.bsz

            # Convert audio length from absolute to percentage
            if key == 'audio_length':
                for x in np.nditer(value, op_flags=['readwrite']):
                    x[...] = x/self.max_duration*100
                value = value.astype(np.uint8, copy=False)

            if key == 'char_map':
                x = value
                x = x[x != 0]
                x = np.reshape(x, (1, -1))
                x = np.ascontiguousarray(x)  # Contigious array needed for GPU backend
            else:
                # Adjust Aeon data layout to Neon layout, e.g. shape (N,C,H,W) -> (N,C*H*W)
                x = np.reshape(value, (reshape_rows, -1))
                x = np.ascontiguousarray(x.T)  # Contigious array needed for GPU backend

            # Patch that forces the numpy array into C order to fix the strides for NumPy > 1.11.1
            x = np.array(x, order='C')

            # Create tensor
            self.outputs[key] = self.be.array(x, dtype=value.dtype)

        return tuple(self.outputs.values())

    def shapes(self):
        # Get tuple of shape values only
        shapes = []
        for name, value in self.dataloader.axes_info:
            vals = ()
            for child_name, child_value in value:
                vals = vals + (child_value, )
            shapes.append(vals)

        return tuple(shapes)
